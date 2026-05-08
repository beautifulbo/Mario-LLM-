"""
Stage 2 复合损失函数
对应论文公式 (9)(10): 性能加权损失 + KL 正则化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Stage2CompositeLoss(nn.Module):
    """
    论文公式 (10):
    L_S2 = (1/|B|) * sum_v [ sum_k q_v^(k) * ell_v^(k) + lambda * KL(q_v || p_v) ]

    其中:
    - ell_v^(k): LLM 对模板 k 的负对数似然损失 (公式 9)
    - q_v = softmax(-[ell_txt, ell_vis, ell_mm]): 性能后验
    - p_v = softmax(s_v): 路由器输出概率
    - lambda: KL 正则化系数
    """

    def __init__(self, kl_weight=0.1):
        super().__init__()
        self.kl_weight = kl_weight

    def forward(self, llm_losses, router_probs):
        """
        Args:
            llm_losses: (batch, 3) 三个模板的 LLM loss [ell_txt, ell_vis, ell_mm]
            router_probs: (batch, 3) 路由器输出概率 p_v = [p_txt, p_vis, p_mm]
        Returns:
            total_loss: 标量
            posterior: (batch, 3) 性能后验 q_v (用于分析)
        """
        # q_v = softmax(-[ell_txt, ell_vis, ell_mm]) (论文公式 9 下方)
        posterior = F.softmax(-llm_losses, dim=-1)  # (batch, 3)

        # 性能加权损失: sum_k q_v^(k) * ell_v^(k)
        weighted_loss = (posterior * llm_losses).sum(dim=-1).mean()  # scalar

        # KL(q_v || p_v)
        # KL = sum q * (log q - log p)
        kl_div = F.kl_div(
            router_probs.log(),
            posterior,
            reduction='batchmean'
        )

        total_loss = weighted_loss + self.kl_weight * kl_div
        return total_loss, posterior
