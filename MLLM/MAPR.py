"""
Modality-Adaptive Prompt Router (MAPR)
对应论文公式 (7)(8): 路由器输入构造 + MLP 路由
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MAPRouter(nn.Module):
    """
    轻量级 MLP 路由器，为每个节点选择最优模态模板。
    输入: z_v = [h_text_v; h_image_v; phi^(1)(v); phi^(2)(v); log(d_v)]  in R^{4d+1}
    输出: s_v in R^3 -> softmax -> p_v = [p_txt, p_vis, p_mm]
    """

    def __init__(self, embed_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        input_dim = 4 * embed_dim + 1  # h_text + h_image + phi1 + phi2 + log_degree

        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),
        )

    def build_router_input(self, h_text, h_image, neighbors_1_indices, neighbors_2_indices,
                           degrees, all_h_text=None, all_h_image=None):
        """
        构造路由器输入向量 z_v，对应论文公式 (7)(8)。

        Args:
            h_text: (embed_dim,) 锚节点的 Stage1 文本特征
            h_image: (embed_dim,) 锚节点的 Stage1 图像特征
            neighbors_1_indices: 1-hop 邻居索引 tensor
            neighbors_2_indices: 2-hop 邻居索引 tensor
            degrees: 节点度数 (int or scalar tensor)
            all_h_text: 全节点文本特征 (num_nodes, embed_dim)，用于查邻居
            all_h_image: 全节点图像特征 (num_nodes, embed_dim)，用于查邻居
        Returns:
            z_v: (4*embed_dim + 1,) 路由器输入
        """
        device = h_text.device

        # phi^(1)(v): 1-hop 邻居的均值池化 (论文公式 8)
        if len(neighbors_1_indices) > 0 and all_h_text is not None:
            n1_text = all_h_text[neighbors_1_indices]  # (K1, d)
            n1_image = all_h_image[neighbors_1_indices]
            phi1 = (n1_text + n1_image).mean(dim=0)  # (d,)
        else:
            phi1 = torch.zeros_like(h_text)

        # phi^(2)(v): 2-hop 邻居的均值池化
        if len(neighbors_2_indices) > 0 and all_h_text is not None:
            n2_text = all_h_text[neighbors_2_indices]
            n2_image = all_h_image[neighbors_2_indices]
            phi2 = (n2_text + n2_image).mean(dim=0)
        else:
            phi2 = torch.zeros_like(h_text)

        # log(d_v)
        log_degree = torch.tensor(
            [torch.log(torch.tensor(float(degrees) + 1.0))],
            device=device, dtype=h_text.dtype
        )

        # z_v = [h_text; h_image; phi1; phi2; log_d]
        z_v = torch.cat([h_text, h_image, phi1, phi2, log_degree], dim=0)
        return z_v

    def forward(self, z):
        """
        Args:
            z: (batch, 4*embed_dim+1) 路由器输入
        Returns:
            probs: (batch, 3) 路由概率 [p_txt, p_vis, p_mm]
            logits: (batch, 3) 路由 logits
        """
        logits = self.router(z)
        probs = F.softmax(logits, dim=-1)
        return probs, logits
