import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import nn
import time
import dgl

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, graph, buckets_num=6,global_dist_matrix=None):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.graph = graph
        self.buckets_num = buckets_num
        self.dist_matrix = global_dist_matrix.cpu()  # 使用全局距离矩阵
        
        # 每个头独立的桶参数
        self.buckets = nn.ParameterList([
            nn.Parameter(torch.randn(buckets_num)) for _ in range(num_heads)
        ])
        
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_linear_1 = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_linear_2 = nn.Linear(embed_dim, embed_dim, bias=True)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(embed_dim)

    @torch.compiler.disable  # 禁止编译以避免潜在的兼容性问题
    def _compute_position_embedding_batch(self, idx, device):
        """
        只计算一次子图和最短路径，并为所有头并行生成位置编码矩阵
        Returns: (num_heads, seq_len, seq_len)
        """
        # 1. 提取子图并计算最短路径 (仅执行一次)
        idx_list = idx.cpu().tolist()
        
        # 2. 清洗距离矩阵，将 -1 (不可达) 映射为 buckets_num - 1
        dist_matrix = self.dist_matrix[idx_list][:, idx_list]
        dist_matrix = dist_matrix.clamp(min=0, max=self.buckets_num - 1).long()

        dist_matrix = dist_matrix.to(device,non_blocking=True)  # (seq_len, seq_len)
        
        # 3. 并行查表生成所有头的位置编码
        # buckets_stacked: (num_heads, buckets_num)
        buckets_stacked = torch.stack([b for b in self.buckets], dim=0)
        
        # dist_matrix: (seq_len, seq_len) -> 扩展到 (num_heads, seq_len, seq_len)
        dist_expanded = dist_matrix.unsqueeze(0).expand(self.num_heads, -1, -1)
        
        # 高级索引查表，瞬间生成所有头的偏置矩阵
        # pos_emb 形状: (num_heads, seq_len, seq_len)
        pos_emb = buckets_stacked.gather(1, dist_expanded.reshape(self.num_heads, -1)).reshape(self.num_heads, -1, dist_matrix.size(1))
        
        return pos_emb.to(device)

    def forward(self, x, idx):
        batch_size, seq_len, _ = x.size()
        
        # ✅ 在多头计算前，一次性生成位置编码 (num_heads, seq_len, seq_len)
        pos_emb = self._compute_position_embedding_batch(idx, x.device)
        
        # 线性变换并分头
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, S, D)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # ✅ 批量矩阵乘法，消除 for head 循环
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # 加入位置编码: pos_emb 需要在 batch 维度广播 (1, H, S, S)
        scores = scores + pos_emb.unsqueeze(0)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v) # (B, H, S, D)
        
        # 合并头部
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        output = self.out_linear_1(attn_output)
        output = self.gelu(output)
        output = self.out_linear_2(output)
        output = self.ln(output+x)


        # 你原代码的逻辑：只更新第0个token的特征，其余保持原样
        h = output[:, 0:1, :]  # (batchsize, 1, embed_dim) 保持 3D
        x1 = x[:, 1:, :]
        output = torch.cat([h, x1], dim=1)
        
        return output


class RepeatedMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads,graph,buckets_num=6,layers=6, global_dist_matrix=None):
        super(RepeatedMultiHeadAttention, self).__init__()
        self.layers = layers
        self.attention=nn.ModuleList([MultiHeadAttention(embed_dim, num_heads,graph,buckets_num,global_dist_matrix) for _ in range(layers)])

    def forward(self, x,idx):
        for i in range(self.layers):
            x = self.attention[i](x,idx)
        return x

class FeatureExtractor(nn.Module):
    def __init__(self, embed_dim, num_heads,graph,buckets_num=6,layers=6, global_dist_matrix=None):
        super(FeatureExtractor, self).__init__()
        self.repeated_attention = RepeatedMultiHeadAttention(embed_dim, num_heads,graph,buckets_num,layers, global_dist_matrix)

    def forward(self, x,idx):
        return self.repeated_attention(x,idx)
    
class MarioFeatureExtractor(nn.Module):
    def __init__(self, embed_dim, num_heads,graph,buckets_num=6,layers=6, dist_matrix_path="../Data/Reddit/RedditDist.pt"):
        super(MarioFeatureExtractor, self).__init__()

        #  在构建网络前，一次性加载全局矩阵到内存
        print(f"Loading global distance matrix from {dist_matrix_path} ...")
        global_dist_matrix = torch.load(dist_matrix_path, mmap=True, weights_only=True)
        print(f"Loaded! Shape: {global_dist_matrix.shape}")

        self.text_feature_extractor = FeatureExtractor(embed_dim, num_heads,graph,buckets_num,layers, global_dist_matrix)
        self.image_feature_extractor = FeatureExtractor(embed_dim, num_heads,graph,buckets_num,layers, global_dist_matrix)

    def forward(self, text, image,idx):
        text_features = self.text_feature_extractor(text,idx)
        image_features = self.image_feature_extractor(image,idx)
        return text_features, image_features

# 以上是生成器的代码，下面是Stage2的代码  

class SpecialTokenProjector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SpecialTokenProjector, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
    
    
class PromptTemplate(object):
    def __init__(self, path,text_nodefeatures_path=None,image_nodefeatures_path=None,center=None,choose_modal="text",neighbor_hop=None):
        if path is None:
            raise ValueError("请提供图的路径以初始化 PromptTemplate")
        if text_nodefeatures_path is None or image_nodefeatures_path is None:
            raise ValueError("请提供文本和图像节点特征的路径以初始化 PromptTemplate")
        if center is None:
            raise ValueError("请提供中心节点的 ID 以初始化 PromptTemplate")
        self.graph=dgl.load_graphs(path)[0][0]
        self.special_token_projector = SpecialTokenProjector(input_dim=768, output_dim=768)
        self.text_nodefeatures=torch.load(text_nodefeatures_path,mmap=True)  # 加载文本节点特征
        self.image_nodefeatures=torch.load(image_nodefeatures_path,mmap=True)  # 加载图像节点特征
        self.center=center  # 中心节点的 ID
        self.hops_1_top_k_neighbors = self.get_top_k_hops_1_mario(self.graph, self.center, k=5)  # 获取第一跳的 Top-K 邻居
        self.hops_2_top_k_neighbors = self.get_top_k_hops_2_mario(self.graph, self.center, k=5)  # 获取第二跳的 Top-K 邻居
        self.template=self.generate_template(modal=choose_modal,neighbor_hop_number=neighbor_hop)  # 生成提示模板
    def get_top_k_hops_1_mario(self , g, center_node, k):
        """Mario 论文公式中的 Top-K 邻居选择"""
        neighbors = g.successors(center_node)
        
        if len(neighbors) <= k:
            return neighbors

        # 拼接文本和图像特征，对应论文中的 [h^text_u || h^image_u]
        center_feat = torch.cat([
            self.text_nodefeatures[0, center_node],
            self.image_nodefeatures[0, center_node]
        ], dim=-1).squeeze(0)  # (1, 2*dim)

        neighbor_feats = torch.cat([
            self.text_nodefeatures[0, neighbors],
            self.image_nodefeatures[0, neighbors]
        ], dim=-1).squeeze(0)  # (num_neighbors, 2*dim)

        # 计算余弦相似度
        sim_scores = F.cosine_similarity(center_feat, neighbor_feats, dim=1)
        
        # 取 Top-K
        _, topk_indices = torch.topk(sim_scores, k)
        top_k_neighbors = neighbors[topk_indices]
        
        return top_k_neighbors

    
    def get_top_k_hops_2_mario(self, g, center_node, k):
        """Mario 论文公式中的 2-hop Top-K 邻居选择"""
        neighbors_1 = g.successors(center_node)
        neighbor_set = set()
        for n in neighbors_1:
            n = n.item() if isinstance(n, torch.Tensor) else n
            if n != center_node:
                hops_2 = g.successors(n)
                for h in hops_2:
                    h = h.item() if isinstance(h, torch.Tensor) else h
                    if h != center_node:
                        neighbor_set.add(h)

        neighbors = torch.tensor(list(neighbor_set), dtype=torch.long)

        if len(neighbors) <= k:
            return neighbors

        # 拼接文本和图像特征，对应论文中的 [h^text_u || h^image_u]
        center_feat = torch.cat([
            self.text_nodefeatures[0, center_node],
            self.image_nodefeatures[0, center_node]
        ], dim=-1).squeeze(0)  # (2*dim,)

        neighbor_feats = torch.cat([
            self.text_nodefeatures[0, neighbors],
            self.image_nodefeatures[0, neighbors]
        ], dim=-1).squeeze(0)  # (num_neighbors, 2*dim)

        # 计算余弦相似度
        sim_scores = F.cosine_similarity(center_feat.unsqueeze(0), neighbor_feats, dim=1)

        # 取 Top-K
        _, topk_indices = torch.topk(sim_scores, k)
        top_k_neighbors = neighbors[topk_indices]

        return top_k_neighbors
    
    def generate_template(self, modal="text", neighbor_hop_number=None):
        """Generate prompt template with task description, raw text, and graph special tokens."""
        if neighbor_hop_number is None:
            neighbors = set(self.hops_1_top_k_neighbors.tolist()).union(
                set(self.hops_2_top_k_neighbors.tolist()))
        elif neighbor_hop_number == 1:
            neighbors = set(self.hops_1_top_k_neighbors.tolist())
        elif neighbor_hop_number == 2:
            neighbors = set(self.hops_2_top_k_neighbors.tolist())
        else:
            raise ValueError("neighbor_hop_number must be None, 1 or 2")

        prompt_parts = []

        if modal == "text":
            prompt_parts.append(
                self.special_token_projector(self.text_nodefeatures[0, self.center]).squeeze(0))
            for n in neighbors:
                prompt_parts.append(
                    self.special_token_projector(self.text_nodefeatures[0, n]).squeeze(0))

        elif modal == "image":
            prompt_parts.append(
                self.special_token_projector(self.image_nodefeatures[0, self.center]).squeeze(0))
            for n in neighbors:
                prompt_parts.append(
                    self.special_token_projector(self.image_nodefeatures[0, n]).squeeze(0))

        elif modal == "both":
            prompt_parts.append(
                self.special_token_projector(self.text_nodefeatures[0, self.center]).squeeze(0))
            prompt_parts.append(
                self.special_token_projector(self.image_nodefeatures[0, self.center]).squeeze(0))
            for n in neighbors:
                prompt_parts.append(
                    self.special_token_projector(self.text_nodefeatures[0, n]).squeeze(0))
                prompt_parts.append(
                    self.special_token_projector(self.image_nodefeatures[0, n]).squeeze(0))
        else:
            raise ValueError("modal must be 'text', 'image' or 'both'")

        return torch.stack(prompt_parts, dim=0)

    @classmethod
    def return_prompt_template(cls, path, text_nodefeatures_path, image_nodefeatures_path,
                               center, choose_modal="text", neighbor_hop=None,
                               task_description="nc", csv_path=None):
        """Factory method: create instance and return prompt template."""
        instance = cls(path, text_nodefeatures_path, image_nodefeatures_path,
                       center, choose_modal, neighbor_hop, task_description, csv_path)
        return instance


from TaskPrompt import NC_TASK, LP_TASK


class MarioPromptTemplate(PromptTemplate):
    def __init__(self, path, text_nodefeatures_path=None, image_nodefeatures_path=None,
                 center=None, choose_modal="text", neighbor_hop=None,
                 task_description=None, csv_path=None):
        super(MarioPromptTemplate, self).__init__(path, text_nodefeatures_path,
                                                  image_nodefeatures_path, center,
                                                  choose_modal, neighbor_hop)
        if task_description is None:
            raise ValueError("请提供任务描述类型以初始化 MarioPromptTemplate")
        if task_description == "nc":
            self.task_description = NC_TASK
        elif task_description == "lp":
            self.task_description = LP_TASK
        else:
            raise ValueError("task_description 必须是 'nc' 或 'lp'")

        self.csv_path = csv_path
        if choose_modal not in ["text", "image", "both"]:
            raise ValueError("choose_modal 参数必须是 'text'、'image' 或 'both'")

        import pandas as pd
        if self.csv_path is not None:
            self.csv_data = pd.read_csv(self.csv_path)
        else:
            raise ValueError("请提供 CSV 文件路径以初始化 MarioPromptTemplate")
        self.raw_caption = self.csv_data.loc[
            self.csv_data['id'] == self.center, 'caption'].values[0]

    def build_full_prompt(self, modal="text", neighbor_hop_number=None):
        """构造完整 prompt: task_description + raw_caption + special tokens"""
        special_tokens = self.generate_template(modal=modal,
                                                neighbor_hop_number=neighbor_hop_number)
        return {
            "task_description": self.task_description,
            "raw_caption": self.raw_caption,
            "special_tokens": special_tokens,
        }
