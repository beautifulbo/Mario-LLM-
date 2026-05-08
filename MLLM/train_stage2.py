"""
Stage 2: Modality-Adaptive Graph Instruction Tuning
对应论文 Section 4.2: MAPR 路由器 + LLM LoRA 微调 + 复合损失

训练策略 (论文 Section 4.3):
  (i)  冻结 Stage1 参数 Theta_S1*
  (ii) 用 LoRA 微调 LLM，联合训练 MAPR，使用复合损失 L_S2
  (iii) 推理时 MAPR 切换为 hard routing
"""
import os
import argparse
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
import pandas as pd
import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from Mario import MarioFeatureExtractor, SpecialTokenProjector
from MAPR import MAPRouter
from loss_stage2 import Stage2CompositeLoss
from train import normalize_state_dict_keys


# ============================================================
#  Prompt Builder
# ============================================================

def build_prompt_inputs(tokenizer, task_description, raw_caption, special_token_embeds,
                        answer_text, llm_embed_layer, device):
    """
    构造 LLM 输入: [task_tokens; caption_tokens; special_graph_tokens] + [answer_tokens]

    Args:
        tokenizer: LLM tokenizer
        task_description: 任务描述文本
        raw_caption: 节点原始文本
        special_token_embeds: (num_graph_tokens, embed_dim) 图特殊 token 嵌入
        answer_text: 答案文本 (如标签名)
        llm_embed_layer: LLM 的 embedding 层
        device: 设备
    Returns:
        inputs_embeds: (1, total_seq_len, hidden_dim)
        labels: (1, total_seq_len) prompt 部分为 -100
        prompt_len: int, prompt token 数量
    """
    prompt_str = f"{task_description}\n\nNode content: {raw_caption}\n\nGraph context tokens: "
    answer_str = f" {answer_text}{tokenizer.eos_token}"

    prompt_ids = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    answer_ids = tokenizer(answer_str, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    prompt_embeds = llm_embed_layer(prompt_ids)  # (1, prompt_len, hidden_dim)
    answer_embeds = llm_embed_layer(answer_ids)  # (1, answer_len, hidden_dim)

    # 拼接: [prompt_embeds; special_token_embeds; answer_embeds]
    graph_tokens = special_token_embeds.unsqueeze(0)  # (1, G, hidden_dim)
    inputs_embeds = torch.cat([prompt_embeds, graph_tokens, answer_embeds], dim=1)

    prompt_len = prompt_ids.size(1) + special_token_embeds.size(0)

    # labels: prompt 部分用 -100 遮蔽，只在 answer 部分计算 loss
    labels = torch.full((1, inputs_embeds.size(1)), -100, dtype=torch.long, device=device)
    labels[0, prompt_len:] = answer_ids[0]

    return inputs_embeds, labels, prompt_len


def compute_llm_loss(llm_model, inputs_embeds, labels):
    """计算 LLM 的 causal LM loss"""
    outputs = llm_model(inputs_embeds=inputs_embeds, labels=labels)
    return outputs.loss


# ============================================================
#  Stage 2 Dataset
# ============================================================

class Stage2Dataset:
    """
    管理 Stage 2 训练所需的所有数据:
    - 图结构 (用于获取邻居)
    - Stage 1 特征 (冻结)
    - 节点 caption 和 label (来自 CSV)
    - 预计算的 Top-K 邻居
    """

    def __init__(self, graph_path, text_feat_path, image_feat_path,
                 csv_path, label_col="label", caption_col="caption",
                 id_col="id", k_neighbors=5):
        self.graph = dgl.load_graphs(graph_path)[0][0]
        self.text_features = torch.from_numpy(np.load(text_feat_path)).float()
        self.image_features = torch.from_numpy(np.load(image_feat_path)).float()
        self.num_nodes = self.text_features.size(0)

        self.csv_data = pd.read_csv(csv_path)
        self.label_col = label_col
        self.caption_col = caption_col
        self.id_col = id_col
        self.k = k_neighbors

        # 预计算所有节点的 Top-K 邻居
        self.hop1_neighbors = {}
        self.hop2_neighbors = {}
        self.degrees = {}

        print("Precomputing Top-K neighbors for all nodes...")
        for node_id in tqdm.tqdm(range(self.num_nodes)):
            self._compute_neighbors(node_id)

    def _compute_neighbors(self, node_id):
        """预计算节点的 1-hop 和 2-hop Top-K 邻居"""
        g = self.graph

        # 1-hop
        succ = g.successors(node_id)
        if len(succ) > self.k:
            sim = self._neighbor_similarity(node_id, succ)
            _, idx = torch.topk(sim, self.k)
            succ = succ[idx]
        self.hop1_neighbors[node_id] = succ
        self.degrees[node_id] = len(g.successors(node_id))

        # 2-hop
        hop2_set = set()
        for n in self.hop1_neighbors[node_id]:
            n_item = n.item() if isinstance(n, torch.Tensor) else n
            for h in g.successors(n_item):
                h_item = h.item() if isinstance(h, torch.Tensor) else h
                if h_item != node_id:
                    hop2_set.add(h_item)
        hop2 = torch.tensor(list(hop2_set), dtype=torch.long)
        if len(hop2) > self.k:
            sim = self._neighbor_similarity(node_id, hop2)
            _, idx = torch.topk(sim, self.k)
            hop2 = hop2[idx]
        self.hop2_neighbors[node_id] = hop2

    def _neighbor_similarity(self, center, candidates):
        """计算中心节点与候选邻居的余弦相似度 (拼接文本+图像)"""
        center_feat = torch.cat([
            self.text_features[center], self.image_features[center]
        ], dim=-1).unsqueeze(0)
        cand_feats = torch.cat([
            self.text_features[candidates], self.image_features[candidates]
        ], dim=-1)
        return F.cosine_similarity(center_feat, cand_feats, dim=1)

    def get_node_data(self, node_id):
        """获取单个节点的所有 Stage 2 所需数据"""
        row = self.csv_data[self.csv_data[self.id_col] == node_id].iloc[0]
        return {
            "node_id": node_id,
            "caption": row[self.caption_col],
            "label": str(row[self.label_col]),
            "hop1": self.hop1_neighbors[node_id],
            "hop2": self.hop2_neighbors[node_id],
            "degree": self.degrees[node_id],
        }


# ============================================================
#  Stage 2 Trainer
# ============================================================

class Stage2Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. 加载数据集
        print("Loading dataset...")
        self.dataset = Stage2Dataset(
            graph_path=args.graph_path,
            text_feat_path=args.text_data_path,
            image_feat_path=args.image_data_path,
            csv_path=args.csv_path,
            label_col=args.label_col,
            k_neighbors=args.k_neighbors,
        )

        # 2. 加载 SpecialTokenProjector (从 Stage1 checkpoint 或新建)
        embed_dim = self.dataset.text_features.size(-1)
        self.projector = SpecialTokenProjector(
            input_dim=embed_dim, output_dim=args.llm_hidden_dim
        ).to(self.device)

        # 3. 加载 LLM + LoRA
        print(f"Loading LLM from {args.llm_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            args.llm_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        # 冻结 LLM 主干，只训练 LoRA
        for param in self.llm.parameters():
            param.requires_grad = False

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules.split(","),
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(self.llm, lora_config)
        self.llm.print_trainable_parameters()

        self.llm_embed_layer = self.llm.get_input_embeddings()

        # 4. MAPR 路由器
        self.router = MAPRouter(
            embed_dim=embed_dim,
            hidden_dim=args.router_hidden_dim,
            dropout=args.router_dropout,
        ).to(self.device)

        # 5. 复合损失
        self.criterion = Stage2CompositeLoss(kl_weight=args.kl_weight)

        # 6. 优化器: 分组学习率
        self.optimizer = torch.optim.AdamW([
            {"params": self.router.parameters(), "lr": args.router_lr},
            {"params": self.projector.parameters(), "lr": args.projector_lr},
            {"params": [p for p in self.llm.parameters() if p.requires_grad], "lr": args.llm_lr},
        ], weight_decay=args.weight_decay)

        # 任务描述
        from TaskPrompt import NC_TASK, LP_TASK
        self.task_desc = NC_TASK if args.task == "nc" else LP_TASK

        # 可用节点列表 (排除没有 caption 的节点)
        self.train_nodes = list(range(self.dataset.num_nodes))

    def _get_all_neighbors(self, hop1, hop2):
        """合并 1-hop 和 2-hop 邻居，排除重复"""
        all_n = set()
        for t in [hop1, hop2]:
            for n in t:
                all_n.add(n.item() if isinstance(n, torch.Tensor) else n)
        return list(all_n)

    def _build_graph_tokens(self, node_id, modal, hop1, hop2):
        """
        构造图特殊 token 嵌入，对应论文 Prompt Template Bank。
        txt:  {<GT_v>, <GT_u1>, ..., <GT_uK>}
        vis:  {<GI_v>, <GI_u1>, ..., <GI_uK>}
        mm:   {<GT_v>, <GI_v>, ..., <GT_uK>, <GI_uK>}
        """
        text_feat = self.dataset.text_features.to(self.device)
        image_feat = self.dataset.image_features.to(self.device)
        neighbors = self._get_all_neighbors(hop1, hop2)

        tokens = []
        if modal == "text":
            tokens.append(self.projector(text_feat[node_id].unsqueeze(0)))
            for n in neighbors:
                tokens.append(self.projector(text_feat[n].unsqueeze(0)))
        elif modal == "image":
            tokens.append(self.projector(image_feat[node_id].unsqueeze(0)))
            for n in neighbors:
                tokens.append(self.projector(image_feat[n].unsqueeze(0)))
        elif modal == "both":
            tokens.append(self.projector(text_feat[node_id].unsqueeze(0)))
            tokens.append(self.projector(image_feat[node_id].unsqueeze(0)))
            for n in neighbors:
                tokens.append(self.projector(text_feat[n].unsqueeze(0)))
                tokens.append(self.projector(image_feat[n].unsqueeze(0)))

        return torch.cat(tokens, dim=0)  # (num_tokens, llm_hidden_dim)

    def train_step(self, batch_nodes):
        """
        单步训练: 对 batch 中每个节点，计算三个模板的 LLM loss，
        然后用复合损失更新 MAPR + LLM LoRA + Projector。

        Args:
            batch_nodes: list of node_id
        Returns:
            loss_dict: dict of scalar losses
        """
        self.router.train()
        self.projector.train()
        self.llm.train()

        all_router_inputs = []
        all_llm_losses = []

        for node_id in batch_nodes:
            data = self.dataset.get_node_data(node_id)
            caption = data["caption"]
            label = data["label"]
            hop1, hop2 = data["hop1"], data["hop2"]
            degree = data["degree"]

            # 构造 MAPR 输入
            h_text = self.dataset.text_features[node_id].to(self.device)
            h_image = self.dataset.image_features[node_id].to(self.device)
            hop1_idx = [n.item() if isinstance(n, torch.Tensor) else n for n in hop1]
            hop2_idx = [n.item() if isinstance(n, torch.Tensor) else n for n in hop2]

            z_v = self.router.build_router_input(
                h_text, h_image,
                torch.tensor(hop1_idx, dtype=torch.long, device=self.device),
                torch.tensor(hop2_idx, dtype=torch.long, device=self.device),
                degree,
                all_h_text=self.dataset.text_features.to(self.device),
                all_h_image=self.dataset.image_features.to(self.device),
            )
            all_router_inputs.append(z_v)

            # 计算三个模板的 LLM loss
            node_llm_losses = []
            for modal in ["text", "image", "both"]:
                graph_tokens = self._build_graph_tokens(node_id, modal, hop1, hop2)
                inputs_embeds, labels, _ = build_prompt_inputs(
                    self.tokenizer, self.task_desc, caption,
                    graph_tokens, label, self.llm_embed_layer, self.device
                )
                loss = compute_llm_loss(self.llm, inputs_embeds, labels)
                node_llm_losses.append(loss)

            all_llm_losses.append(torch.stack(node_llm_losses))  # (3,)

        # 堆叠 batch 数据
        router_input = torch.stack(all_router_inputs)    # (B, 4d+1)
        llm_losses = torch.stack(all_llm_losses)          # (B, 3)

        # MAPR 前向
        router_probs, _ = self.router(router_input)       # (B, 3)

        # 复合损失
        total_loss, posterior = self.criterion(llm_losses, router_probs)

        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.router.parameters()) +
            list(self.projector.parameters()) +
            [p for p in self.llm.parameters() if p.requires_grad],
            max_norm=1.0
        )
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "weighted_loss": (posterior * llm_losses).sum(dim=-1).mean().item(),
            "llm_loss_txt": llm_losses[:, 0].mean().item(),
            "llm_loss_vis": llm_losses[:, 1].mean().item(),
            "llm_loss_mm": llm_losses[:, 2].mean().item(),
            "router_entropy": -(router_probs * (router_probs + 1e-8).log()).sum(-1).mean().item(),
        }

    @torch.no_grad()
    def evaluate(self, eval_nodes, answer_map=None):
        """
        推理: MAPR hard routing -> 选择最优模板 -> LLM 生成

        Args:
            eval_nodes: list of node_id
            answer_map: dict {node_id: ground_truth_label} 用于计算准确率
        Returns:
            metrics: dict
        """
        self.router.eval()
        self.projector.eval()
        self.llm.eval()

        correct = 0
        total = 0
        modal_counts = {"text": 0, "image": 0, "both": 0}
        modal_names = ["text", "image", "both"]

        for node_id in tqdm.tqdm(eval_nodes, desc="Evaluating"):
            data = self.dataset.get_node_data(node_id)
            caption = data["caption"]
            hop1, hop2 = data["hop1"], data["hop2"]
            degree = data["degree"]

            # MAPR 路由
            h_text = self.dataset.text_features[node_id].to(self.device)
            h_image = self.dataset.image_features[node_id].to(self.device)
            hop1_idx = [n.item() if isinstance(n, torch.Tensor) else n for n in hop1]
            hop2_idx = [n.item() if isinstance(n, torch.Tensor) else n for n in hop2]

            z_v = self.router.build_router_input(
                h_text, h_image,
                torch.tensor(hop1_idx, dtype=torch.long, device=self.device),
                torch.tensor(hop2_idx, dtype=torch.long, device=self.device),
                degree,
                all_h_text=self.dataset.text_features.to(self.device),
                all_h_image=self.dataset.image_features.to(self.device),
            )
            probs, _ = self.router(z_v.unsqueeze(0))
            chosen_modal = modal_names[probs.argmax(dim=-1).item()]
            modal_counts[chosen_modal] += 1

            # 用选定模板构造 prompt 并生成
            graph_tokens = self._build_graph_tokens(node_id, chosen_modal, hop1, hop2)
            prompt_str = f"{self.task_desc}\n\nNode content: {caption}\n\nGraph context tokens: "
            prompt_ids = self.tokenizer(prompt_str, return_tensors="pt",
                                        add_special_tokens=True).input_ids.to(self.device)
            prompt_embeds = self.llm_embed_layer(prompt_ids)
            graph_token_embeds = graph_tokens.unsqueeze(0)
            gen_embeds = torch.cat([prompt_embeds, graph_token_embeds], dim=1)

            output_ids = self.llm.generate(
                inputs_embeds=gen_embeds,
                max_new_tokens=20,
                do_sample=False,
            )
            pred_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

            if answer_map is not None:
                gt = answer_map.get(node_id, "")
                if pred_text.lower().startswith(str(gt).lower()):
                    correct += 1
                total += 1

        metrics = {"modal_distribution": modal_counts}
        if total > 0:
            metrics["accuracy"] = correct / total
        return metrics

    def train(self):
        """完整训练循环"""
        args = self.args
        os.makedirs(args.save_dir, exist_ok=True)

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(args.num_epochs):
            random.shuffle(self.train_nodes)
            epoch_losses = []
            pbar = tqdm.tqdm(
                range(0, len(self.train_nodes), args.batch_size),
                desc=f"Epoch {epoch+1}/{args.num_epochs}"
            )

            for start in pbar:
                batch_nodes = self.train_nodes[start:start + args.batch_size]
                loss_dict = self.train_step(batch_nodes)
                epoch_losses.append(loss_dict)
                pbar.set_postfix({
                    "loss": f"{loss_dict['total_loss']:.4f}",
                    "txt": f"{loss_dict['llm_loss_txt']:.4f}",
                    "vis": f"{loss_dict['llm_loss_vis']:.4f}",
                    "mm": f"{loss_dict['llm_loss_mm']:.4f}",
                })

            # Epoch 统计
            avg_loss = np.mean([d["total_loss"] for d in epoch_losses])
            avg_txt = np.mean([d["llm_loss_txt"] for d in epoch_losses])
            avg_vis = np.mean([d["llm_loss_vis"] for d in epoch_losses])
            avg_mm = np.mean([d["llm_loss_mm"] for d in epoch_losses])
            avg_entropy = np.mean([d["router_entropy"] for d in epoch_losses])

            print(f"\n[Epoch {epoch+1}] loss={avg_loss:.4f}  "
                  f"txt={avg_txt:.4f}  vis={avg_vis:.4f}  mm={avg_mm:.4f}  "
                  f"router_entropy={avg_entropy:.4f}")

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                self._save_checkpoint(epoch, is_best=True)
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            if (epoch + 1) % args.save_every == 0:
                self._save_checkpoint(epoch)

    def _save_checkpoint(self, epoch, is_best=False):
        """保存 Stage 2 checkpoint"""
        save_dir = self.args.save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 保存 MAPR
        torch.save(self.router.state_dict(),
                    os.path.join(save_dir, "mapr_best.pth" if is_best else f"mapr_epoch_{epoch+1}.pth"))
        # 保存 Projector
        torch.save(self.projector.state_dict(),
                    os.path.join(save_dir, "projector_best.pth" if is_best else f"projector_epoch_{epoch+1}.pth"))
        # 保存 LLM LoRA
        tag = "best" if is_best else f"epoch_{epoch+1}"
        self.llm.save_pretrained(os.path.join(save_dir, f"llm_lora_{tag}"))

        if is_best:
            print(f"  >> Best model saved (loss={self.args.save_dir})")


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Mario Stage 2: Modality-Adaptive Graph Instruction Tuning")

    # 数据
    parser.add_argument("--graph_path", type=str, default="../Data/Reddit/RedditSGraph_train.pt")
    parser.add_argument("--text_data_path", type=str, default="../Data/Reddit/TextFeature/RedditS_Qwen2_VL_7B_Instruct_100_mean_train.npy")
    parser.add_argument("--image_data_path", type=str, default="../Data/Reddit/ImageFeature/RedditS_Qwen2-VL-7B_Instruct_visual_train.npy")
    parser.add_argument("--csv_path", type=str, required=True, help="CSV with columns: id, caption, label")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--k_neighbors", type=int, default=5)

    # LLM
    parser.add_argument("--llm_path", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--llm_hidden_dim", type=int, default=4096)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # MAPR
    parser.add_argument("--router_hidden_dim", type=int, default=256)
    parser.add_argument("--router_dropout", type=float, default=0.1)

    # 训练
    parser.add_argument("--task", type=str, default="nc", choices=["nc", "lp"])
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--router_lr", type=float, default=1e-3)
    parser.add_argument("--projector_lr", type=float, default=1e-4)
    parser.add_argument("--llm_lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--kl_weight", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default="../trained_models/stage2")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    trainer = Stage2Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
