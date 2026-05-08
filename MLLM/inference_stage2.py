"""
Stage 2 推理: MAPR hard routing -> 选定模板 -> LLM 生成
对应论文 Section 4.3: 推理时 MAPR 切换为 hard policy
  k* = argmax p_v^(k)
"""
import os
import argparse
import json

import torch
import numpy as np
import dgl
import pandas as pd
import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from MAPR import MAPRouter
from Mario import SpecialTokenProjector
from train_stage2 import Stage2Dataset, build_prompt_inputs
from TaskPrompt import NC_TASK, LP_TASK


class Stage2Inference:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modal_names = ["text", "image", "both"]

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

        embed_dim = self.dataset.text_features.size(-1)

        # 2. 加载 SpecialTokenProjector
        self.projector = SpecialTokenProjector(
            input_dim=embed_dim, output_dim=args.llm_hidden_dim
        ).to(self.device)
        proj_ckpt = os.path.join(args.stage2_dir, "projector_best.pth")
        self.projector.load_state_dict(torch.load(proj_ckpt, map_location=self.device, weights_only=True))
        self.projector.eval()

        # 3. 加载 MAPR
        self.router = MAPRouter(
            embed_dim=embed_dim,
            hidden_dim=args.router_hidden_dim,
            dropout=0.0,
        ).to(self.device)
        router_ckpt = os.path.join(args.stage2_dir, "mapr_best.pth")
        self.router.load_state_dict(torch.load(router_ckpt, map_location=self.device, weights_only=True))
        self.router.eval()

        # 4. 加载 LLM + LoRA
        print(f"Loading LLM from {args.llm_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_llm = AutoModelForCausalLM.from_pretrained(
            args.llm_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        lora_dir = os.path.join(args.stage2_dir, "llm_lora_best")
        self.llm = PeftModel.from_pretrained(base_llm, lora_dir)
        self.llm.eval()

        self.llm_embed_layer = self.llm.get_input_embeddings()

        # 任务描述
        self.task_desc = NC_TASK if args.task == "nc" else LP_TASK

    def _get_all_neighbors(self, hop1, hop2):
        all_n = set()
        for t in [hop1, hop2]:
            for n in t:
                all_n.add(n.item() if isinstance(n, torch.Tensor) else n)
        return list(all_n)

    def _build_graph_tokens(self, node_id, modal, hop1, hop2):
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

        return torch.cat(tokens, dim=0)

    @torch.no_grad()
    def predict(self, node_id):
        """
        对单个节点进行推理。
        Returns:
            pred_text: LLM 生成的答案
            chosen_modal: MAPR 选择的模态
            routing_probs: 路由概率
        """
        data = self.dataset.get_node_data(node_id)
        caption = data["caption"]
        hop1, hop2 = data["hop1"], data["hop2"]
        degree = data["degree"]

        # MAPR hard routing
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
        chosen_idx = probs.argmax(dim=-1).item()
        chosen_modal = self.modal_names[chosen_idx]

        # 构造 prompt
        graph_tokens = self._build_graph_tokens(node_id, chosen_modal, hop1, hop2)
        prompt_str = f"{self.task_desc}\n\nNode content: {caption}\n\nGraph context tokens: "
        prompt_ids = self.tokenizer(prompt_str, return_tensors="pt",
                                    add_special_tokens=True).input_ids.to(self.device)
        prompt_embeds = self.llm_embed_layer(prompt_ids)
        gen_embeds = torch.cat([prompt_embeds, graph_tokens.unsqueeze(0)], dim=1)

        # LLM 生成
        output_ids = self.llm.generate(
            inputs_embeds=gen_embeds,
            max_new_tokens=20,
            do_sample=False,
        )
        pred_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        return pred_text, chosen_modal, probs[0].cpu().tolist()

    @torch.no_grad()
    def run_inference(self, eval_nodes=None):
        """批量推理"""
        if eval_nodes is None:
            eval_nodes = list(range(self.dataset.num_nodes))

        results = []
        modal_counts = {"text": 0, "image": 0, "both": 0}

        for node_id in tqdm.tqdm(eval_nodes, desc="Inference"):
            pred, modal, probs = self.predict(node_id)
            modal_counts[modal] += 1
            results.append({
                "node_id": node_id,
                "prediction": pred,
                "modal": modal,
                "probs": {"text": probs[0], "image": probs[1], "both": probs[2]},
            })

        print(f"\nModal distribution: {modal_counts}")
        return results


def main():
    parser = argparse.ArgumentParser(description="Mario Stage 2 Inference")

    # 数据
    parser.add_argument("--graph_path", type=str, default="../Data/Reddit/RedditSGraph_test.pt")
    parser.add_argument("--text_data_path", type=str, default="../Data/Reddit/TextFeature/RedditS_Qwen2_VL_7B_Instruct_100_mean_test.npy")
    parser.add_argument("--image_data_path", type=str, default="../Data/Reddit/ImageFeature/RedditS_Qwen2-VL-7B_Instruct_visual_test.npy")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--k_neighbors", type=int, default=5)

    # 模型
    parser.add_argument("--llm_path", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--llm_hidden_dim", type=int, default=4096)
    parser.add_argument("--router_hidden_dim", type=int, default=256)
    parser.add_argument("--stage2_dir", type=str, required=True, help="Directory with trained Stage2 checkpoints")
    parser.add_argument("--task", type=str, default="nc", choices=["nc", "lp"])

    # 输出
    parser.add_argument("--output_path", type=str, default="stage2_predictions.json")

    args = parser.parse_args()

    inferencer = Stage2Inference(args)
    results = inferencer.run_inference()

    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Predictions saved to {args.output_path}")


if __name__ == "__main__":
    main()
