"""
INP-Former: Invertible Neural Process with Transformer
用于多视角无监督异常检测

核心思想：
1. 使用 Transformer 编码器建模多视角 patch 特征的空间-视角相关性
2. 使用 Normalizing Flow 学习正常特征的分布
3. 通过负对数似然 (NLL) 进行异常评分
"""
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 1. Normalizing Flow 模块
# ─────────────────────────────────────────────

class ActNorm(nn.Module):
    """Activation Normalization 层：可逆仿射变换"""

    def __init__(self, dim: int):
        super().__init__()
        # log_scale 参数化（保证 scale > 0）
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.initialized = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args: x [B, D]
        Returns: y [B, D], log_det_jacobian [B]
        """
        if not self.initialized and self.training:
            with torch.no_grad():
                # 用第一个 batch 的均值和标准差初始化
                mean = x.mean(dim=0)
                std = x.std(dim=0).clamp(min=1e-2)
                self.bias.data.copy_(mean)
                # log_scale 初始化为 log(std)
                self.log_scale.data.copy_(torch.log(std))
            self.initialized = True

        # 数值安全: 限制 log_scale 在 [-3, 3] 之间, scale ∈ [0.05, 20]
        log_s = torch.clamp(self.log_scale, min=-3.0, max=3.0)
        y = (x - self.bias) * torch.exp(-log_s)
        log_det = (-log_s).sum().expand(x.size(0))
        return y, log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        log_s = torch.clamp(self.log_scale, min=-3.0, max=3.0)
        return y * torch.exp(log_s) + self.bias


class AffineCoupling(nn.Module):
    """
    仿射 Coupling 层（优化版）
    - 更宽的网络 (hidden=256)
    - 残差连接 (x1 → scale/shift 增加跳跃连接)
    """

    def __init__(self, dim: int, hidden_dim: int = 256):
        super().__init__()
        half_dim = dim // 2
        self.net = nn.Sequential(
            nn.Linear(half_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, half_dim * 2),  # scale + shift
        )
        # 初始化最后一层接近零，使初始变换接近恒等
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        D = x.size(-1)
        x1, x2 = x[..., : D // 2], x[..., D // 2:]

        params = self.net(x1)
        log_scale, shift = params.chunk(2, dim=-1)
        log_scale = torch.tanh(log_scale) * 1.0
        shift = torch.clamp(shift, min=-3.0, max=3.0)

        y2 = x2 * torch.exp(log_scale) + shift
        y = torch.cat([x1, y2], dim=-1)
        log_det = log_scale.sum(dim=-1)
        return y, log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        D = y.size(-1)
        y1, y2 = y[..., : D // 2], y[..., D // 2:]
        params = self.net(y1)
        log_scale, shift = params.chunk(2, dim=-1)
        log_scale = torch.tanh(log_scale) * 1.0
        shift = torch.clamp(shift, min=-3.0, max=3.0)
        x2 = (y2 - shift) * torch.exp(-log_scale)
        return torch.cat([y1, x2], dim=-1)


class Invertible1x1Conv(nn.Module):
    """
    可学习的可逆维度混合层
    用 dim//2 个旋转矩阵参数化，每个旋转作用于一对维度 (2i, 2i+1)
    优点:
    - 无需 slogdet（旋转矩阵 det=1，log_det=0）
    - FP16 友好，不会 NaN
    - 表达力强于固定 Permute
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        assert dim % 2 == 0, f"dim 必须为偶数, 得到 {dim}"
        # dim//2 个旋转角度，随机初始化
        self.angles = nn.Parameter(torch.randn(dim // 2) * 0.01)

    def _get_cos_sin(self):
        c = torch.cos(self.angles)  # [D/2]
        s = torch.sin(self.angles)  # [D/2]
        return c, s

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, D] → 对每对维度应用旋转
        c, s = self._get_cos_sin()  # [D/2]
        x_even = x[..., 0::2]  # [B, D/2] 偶数维度
        x_odd = x[..., 1::2]   # [B, D/2] 奇数维度
        y_even = c * x_even - s * x_odd
        y_odd = s * x_even + c * x_odd
        # 交错合并回来
        y = torch.stack([y_even, y_odd], dim=-1).reshape_as(x)
        # 旋转矩阵 det=1, log_det=0
        log_det = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        return y, log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        # 旋转的逆 = 反向旋转
        c, s = self._get_cos_sin()
        y_even = y[..., 0::2]
        y_odd = y[..., 1::2]
        x_even = c * y_even + s * y_odd
        x_odd = -s * y_even + c * y_odd
        return torch.stack([x_even, x_odd], dim=-1).reshape_as(y)


class NormalizingFlow(nn.Module):
    """
    由多个 ActNorm + Invertible1x1Conv + AffineCoupling 组成的 Normalizing Flow
    将复杂分布映射到标准高斯分布

    改进点 (vs 原版):
    - Invertible1x1Conv 替代固定 Permute，可学习的维度混合
    - AffineCoupling 隐藏层加宽到 256，激活函数改 GELU
    """

    def __init__(self, dim: int, n_layers: int = 8, coupling_hidden: int = 256):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.extend([
                ActNorm(dim),
                Invertible1x1Conv(dim),
                AffineCoupling(dim, hidden_dim=coupling_hidden),
            ])
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args: x [B, D] 输入特征
        Returns:
            z [B, D] 变换后的隐变量（应服从标准高斯）
            log_det [B] 雅可比行列式对数
        """
        log_det_total = torch.zeros(x.size(0), device=x.device)
        z = x
        for layer in self.layers:
            z, log_det = layer(z)
            log_det_total += log_det
        return z, log_det_total

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算输入 x 的对数似然
        log p(x) = log p(z) + log |det(dz/dx)|
        """
        z, log_det = self.forward(x)
        # 标准高斯分布的 log p(z)
        log_pz = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * z.size(-1) * math.log(2 * math.pi)
        return log_pz + log_det


# ─────────────────────────────────────────────
# 2. 多视角 Transformer 编码器
# ─────────────────────────────────────────────

class ViewPatchEncoder(nn.Module):
    """
    多视角 Patch Transformer
    同时建模视角内空间关系与视角间对应关系
    """

    def __init__(
        self,
        dinov2_dim: int = 768,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_ff: int = 1024,
        dropout: float = 0.1,
        num_views: int = 5,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_views = num_views
        self.d_model_proj_input = dinov2_dim  # 记录原始 DINOv2 维度，用于判断多尺度

        # 特征投影：DINOv2 dim → d_model
        # 支持单尺度 (dinov2_dim) 或多尺度 (2*dinov2_dim) 输入
        self.proj = nn.Sequential(
            nn.Linear(dinov2_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        # 多尺度融合投影（可选，dinov2_dim*2 → d_model）
        self.multiscale_proj = nn.Sequential(
            nn.Linear(dinov2_dim * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # 视角嵌入
        self.view_embed = nn.Embedding(num_views, d_model)

        # 可学习的位置编码（patch 位置）
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, 1400, d_model))  # 最大 patch 数
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # CLS token 投影
        self.cls_proj = nn.Sequential(
            nn.Linear(dinov2_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # 可学习的 [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(
        self,
        patch_features: torch.Tensor,
        cls_tokens: torch.Tensor,
        num_patches_h: int,
        num_patches_w: int,
        multi_scale_features: Optional[list] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            patch_features: [B, V, N, D_dino] 多视角 patch 特征
            cls_tokens:     [B, V, D_dino]     多视角 CLS token
            num_patches_h:  水平 patch 数
            num_patches_w:  垂直 patch 数

        Returns:
            dict:
                "encoded":   [B, 1 + V*N, d_model] 编码后特征
                "cls_out":   [B, d_model]            全局 CLS 输出
                "view_tokens": [B, V, d_model]      每个视角的汇总 token
        """
        B, V, N, D = patch_features.shape
        device = patch_features.device

        # 投影到 d_model 维度（支持多尺度输入）
        if patch_features.shape[-1] == 2 * self.d_model_proj_input:
            # 多尺度: 用 multiscale_proj
            x = self.multiscale_proj(patch_features)
        else:
            x = self.proj(patch_features)  # [B, V, N, d_model]

        # 添加位置编码
        x = x + self.pos_embed[:, :, :N, :]  # [B, V, N, d_model]

        # 添加视角嵌入
        view_idx = torch.arange(V, device=device)
        view_emb = self.view_embed(view_idx)  # [V, d_model]
        x = x + view_emb.unsqueeze(0).unsqueeze(2)  # [B, V, N, d_model]

        # 展平视角和 patch 维度：[B, V*N, d_model]
        x = x.reshape(B, V * N, self.d_model)

        # 拼接 CLS token
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        x = torch.cat([cls, x], dim=1)  # [B, 1 + V*N, d_model]

        # Transformer 编码
        encoded = self.transformer(x)  # [B, 1 + V*N, d_model]

        # 提取 CLS 输出
        cls_out = encoded[:, 0, :]  # [B, d_model]

        # 提取每个视角的汇总 token（取每个视角第一个 patch 位置）
        view_tokens = []
        for v in range(V):
            # 视角 v 的 token 范围：[1 + v*N, 1 + (v+1)*N)
            v_tokens = encoded[:, 1 + v * N: 1 + (v + 1) * N, :]
            view_tokens.append(v_tokens.mean(dim=1, keepdim=True))  # [B, 1, d_model]
        view_tokens = torch.cat(view_tokens, dim=1)  # [B, V, d_model]

        return {
            "encoded": encoded,
            "cls_out": cls_out,
            "view_tokens": view_tokens,
        }


# ─────────────────────────────────────────────
# 3. INP-Former 完整模型
# ─────────────────────────────────────────────

class INPFormer(nn.Module):
    """
    INP-Former: 结合多视角 Transformer + Normalizing Flow 的异常检测模型

    训练流程：
    1. DINOv2 提取多视角 patch 特征（冻结）
    2. ViewPatchEncoder 编码多视角特征
    3. NormalizingFlow 学习正常特征分布
    4. 通过 NLL 损失优化

    推理流程：
    1. 提取并编码特征
    2. 通过 NormalizingFlow 计算 NLL 作为异常得分
    """

    def __init__(
        self,
        dinov2_dim: int = 768,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_ff: int = 1024,
        dropout: float = 0.1,
        num_views: int = 5,
        n_flow_layers: int = 6,
        coupling_hidden: int = 128,
        score_type: str = "nll",
    ):
        super().__init__()
        self.score_type = score_type
        self.d_model = d_model

        # 多视角 Transformer 编码器
        self.view_encoder = ViewPatchEncoder(
            dinov2_dim=dinov2_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_ff=dim_ff,
            dropout=dropout,
            num_views=num_views,
        )

        # Transformer 输出后的 LayerNorm（稳定 Flow 输入）
        self.cls_norm = nn.LayerNorm(d_model)
        self.view_norm = nn.LayerNorm(d_model)

        # Normalizing Flow（对 CLS 特征建模）
        self.flow_cls = NormalizingFlow(
            dim=d_model,
            n_layers=n_flow_layers,
            coupling_hidden=coupling_hidden,
        )

        # Normalizing Flow（对每个视角 token 建模）
        self.flow_view = NormalizingFlow(
            dim=d_model,
            n_layers=n_flow_layers,
            coupling_hidden=coupling_hidden,
        )

        # 特征聚合头：用于将 patch-level 特征聚合到 patch map 用于异常定位
        self.patch_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        dinov2_features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            dinov2_features: DINOv2Extractor.extract_multi_view() 的输出
                - "cls_tokens":     [B, V, D_dino]
                - "patch_features": [B, V, N, D_dino]
                - "multi_scale_features": List[[B, V, N, D_dino]]  (可选)

        Returns:
            dict:
                "nll_cls":    [B]        CLS token 的负对数似然
                "nll_view":   [B, V]     视角 token 的负对数似然
                "patch_map":  [B, V, N, d_model]  patch-level 异常特征图
                "z_cls":      [B, d_model]        CLS 的隐变量
                "z_view":     [B, V, d_model]     视角的隐变量
        """
        # 多尺度特征融合: 取倒数两层拼接
        patch_features = dinov2_features["patch_features"]
        ms_feats = dinov2_features.get("multi_scale_features", None)
        if ms_feats is not None and len(ms_feats) >= 2:
            layer_a = ms_feats[-2]  # 倒数第二层 (e.g. layer 9)
            layer_b = ms_feats[-1]  # 最后一层 (e.g. layer 11, == patch_features)
            patch_features = torch.cat([layer_a, layer_b], dim=-1)  # [B, V, N, 2D]

        # 1. Transformer 编码
        enc_out = self.view_encoder(
            patch_features=patch_features,
            cls_tokens=dinov2_features["cls_tokens"],
            num_patches_h=dinov2_features["num_patches_h"],
            num_patches_w=dinov2_features["num_patches_w"],
        )

        cls_out = enc_out["cls_out"]        # [B, d_model]
        view_tokens = enc_out["view_tokens"]  # [B, V, d_model]

        # 稳定 Flow 输入（LayerNorm 归一化）
        cls_out = self.cls_norm(cls_out)
        view_tokens = self.view_norm(view_tokens)

        # 2. Normalizing Flow: CLS 特征
        z_cls, log_det_cls = self.flow_cls(cls_out)
        # 用 z 空间距离作为训练信号（去掉 log_det，防止 Flow 通过膨胀 log_det 作弊）
        # 推理时 anomaly_score 也用 z 空间 L2 距离，训练和推理保持一致
        nll_cls = 0.5 * (z_cls ** 2).sum(dim=-1)  # [B]

        # 3. Normalizing Flow: 视角特征
        B, V, D = view_tokens.shape
        view_flat = view_tokens.reshape(B * V, D)
        z_view_flat, log_det_view_flat = self.flow_view(view_flat)
        nll_view_flat = 0.5 * (z_view_flat ** 2).sum(dim=-1)
        nll_view = nll_view_flat.reshape(B, V)  # [B, V]

        z_view = z_view_flat.reshape(B, V, D)

        # 4. Patch-level 特征图（用于异常定位）
        encoded = enc_out["encoded"]  # [B, 1 + V*N, d_model]
        # 去掉 CLS token
        patch_encoded = encoded[:, 1:, :]  # [B, V*N, d_model]
        patch_map = self.patch_head(patch_encoded)  # [B, V*N, d_model]
        B_total = patch_map.size(0)
        N = patch_map.size(1) // V
        patch_map = patch_map.reshape(B_total, V, N, self.d_model)

        return {
            "nll_cls": nll_cls,
            "nll_view": nll_view,
            "patch_map": patch_map,
            "z_cls": z_cls,
            "z_view": z_view,
            "log_det_cls": log_det_cls,
            "log_det_view": log_det_view_flat.reshape(B, V),
        }

    def anomaly_score(
        self, dinov2_features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        计算异常得分（推理时使用）

        使用 z 空间 L2 距离作为图像级得分（比 NLL 更鲁棒）。
        正常样本的 z 应接近标准高斯原点，异常样本偏离原点。

        Returns:
            dict:
                "image_score": [B]     图像级异常得分
                "view_score":  [B, V]  视角级异常得分
                "patch_score": [B, V, N]  patch 级异常得分（用于异常定位热力图）
        """
        out = self.forward(dinov2_features)

        # 图像级得分 = z 空间 L2 距离（CLS + 视角均值）
        z_cls = out["z_cls"]      # [B, D]
        z_view = out["z_view"]    # [B, V, D]
        cls_dist = torch.norm(z_cls, dim=-1)                       # [B]
        view_dist = torch.norm(z_view, dim=-1).mean(dim=-1)        # [B]
        image_score = cls_dist + view_dist

        # 视角级得分
        view_score = torch.norm(z_view, dim=-1)  # [B, V]

        # Patch 级得分：patch_head 特征与 CLS 特征的余弦距离
        patch_map = out["patch_map"]  # [B, V, N, D]
        cls_out = z_cls.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, D]
        cos_sim = F.cosine_similarity(
            patch_map, cls_out.expand_as(patch_map), dim=-1
        )  # [B, V, N]
        patch_score = 1.0 - cos_sim  # 越大越异常

        return {
            "image_score": image_score,
            "view_score": view_score,
            "patch_score": patch_score,
        }
