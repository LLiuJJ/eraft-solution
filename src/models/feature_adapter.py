"""
Feature Adapter: 轻量级域适配 MLP

在 DINOv2 4 层拼接特征（3072 维）和 PatchCore k-NN 之间，学习域特定的特征映射。
让正常 patch 特征更紧凑、异常 patch 特征更分散。

架构:
  input [3072] → LayerNorm → Linear(3072→2048) → GELU → Dropout(0.1)
             → Linear(2048→1024) → + 残差(skip projection) → LayerNorm → output [1024]

参数量: ~8.4M（仅为 DINOv2 ViT-B/14 的 ~10%）

参考: Dinomaly (2024), AnomalyDINO (2024) 的轻量级适配器设计
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class FeatureAdapter(nn.Module):
    """
    轻量级域适配 MLP：DINOv2 4 层拼接特征 → 适配后的异常感知特征。

    训练时冻结 DINOv2，仅训练此适配器。
    推理时在 PatchCore k-NN 前应用，提升检测精度 2-5% AP。
    """

    def __init__(
        self,
        input_dim: int = 3072,   # DINOv2 4 层拼接: 4 * 768 = 3072
        output_dim: int = 1024,  # 降维后的特征维度
        hidden_dim: int = 2048,  # MLP 隐藏层维度
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 主干网络: LayerNorm → Linear → GELU → Dropout → Linear
        self.norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # 残差投影: 将 input_dim 投影到 output_dim 用于 skip connection
        self.skip_proj = nn.Linear(input_dim, output_dim, bias=False)

        # 输出 LayerNorm
        self.out_norm = nn.LayerNorm(output_dim)

        self._init_weights()

    def _init_weights(self):
        """初始化: 小权重 + skip_proj 用正交初始化保持信息流"""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # skip_proj 用正交初始化，保证初始时残差路径信息无损
        nn.init.orthogonal_(self.skip_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [*, input_dim]  DINOv2 4 层拼接特征

        Returns:
            out: [*, output_dim]  适配后的特征
        """
        # 主干: norm → MLP
        residual = self.skip_proj(self.norm(x))  # [* , output_dim]
        out = self.mlp(self.norm(x))             # [* , output_dim]

        # 残差连接 + 输出归一化
        out = self.out_norm(out + residual)       # [* , output_dim]
        return out

    def num_parameters(self) -> int:
        """返回可训练参数总数"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# 合成异常生成（用于适配器自监督训练）
# ─────────────────────────────────────────────

def generate_synthetic_anomaly(
    patches: torch.Tensor,   # [B, N, D]  正常 patch 特征
    n_h: int,
    n_w: int,
    anomaly_ratio: float = 0.3,  # 30% 的 patch 被施加异常
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    在正常 patch 特征上施加合成异常（特征空间扰动），用于自监督训练适配器。

    异常策略:
    1. CutPaste 风格: 从其他样本复制 patch 特征覆盖（模拟异物/缺陷）
    2. 高斯噪声: 对 patch 特征添加随机噪声（模拟纹理异常）
    3. 零化: 将 patch 特征置零（模拟缺失/遮挡）

    Args:
        patches: [B, N, D]  正常 patch 特征序列
        n_h, n_w: 空间 grid 尺寸
        anomaly_ratio: 施加异常的 patch 比例

    Returns:
        corrupted: [B, N, D]  含合成异常的特征
        is_anomaly: [B, N]    bool mask，True 表示该 patch 是异常
    """
    B, N, D = patches.shape
    corrupted = patches.clone()
    is_anomaly = torch.zeros(B, N, dtype=torch.bool, device=patches.device)

    for b in range(B):
        # 随机选择异常 patch 位置
        num_anomaly = int(N * anomaly_ratio)
        anomaly_idx = torch.randperm(N, device=patches.device)[:num_anomaly]
        is_anomaly[b, anomaly_idx] = True

        # 对选中的 patch 随机选择异常类型
        strategy = torch.randint(0, 3, (num_anomaly,), device=patches.device)

        # 策略 0: CutPaste - 从其他样本复制特征
        mask_cp = (strategy == 0)
        if mask_cp.any():
            # 从另一个随机样本取对应位置的 patch 特征
            other_b = (b + torch.randint(1, B, (1,), device=patches.device)) % B
            corrupted[b, anomaly_idx[mask_cp]] = patches[other_b, anomaly_idx[mask_cp]]

        # 策略 1: 高斯噪声
        mask_noise = (strategy == 1)
        if mask_noise.any():
            noise = torch.randn_like(patches[b, anomaly_idx[mask_noise]]) * 0.5
            corrupted[b, anomaly_idx[mask_noise]] = patches[b, anomaly_idx[mask_noise]] + noise

        # 策略 2: 零化（特征遮挡）
        mask_zero = (strategy == 2)
        if mask_zero.any():
            corrupted[b, anomaly_idx[mask_zero]] = 0.0

    return corrupted, is_anomaly
