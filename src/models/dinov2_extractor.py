"""
DINOv2 特征提取器（纯 PyTorch 实现）
使用预训练 DINOv2 ViT-B/14 提取多尺度 patch-level 特征

支持两种加载方式：
  1. 本地 checkpoint 路径（推荐，不依赖网络）
  2. torch.hub 在线下载（需要网络 + GitHub 不限流）
"""
import os
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 纯 PyTorch DINOv2 ViT 实现
# ─────────────────────────────────────────────

class LayerScale(nn.Module):
    """DINOv2 使用的 LayerScale 机制"""

    def __init__(self, dim: float, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class FusedQKVAttention(nn.Module):
    """DINOv2 风格的融合 QKV 注意力"""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 融合 Q/K/V 投影
        self.qkv = nn.Linear(dim, dim * 3)
        # 输出投影
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # 融合 QKV → reshape → 多头
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]
        q, k, v = qkv.unbind(0)

        # Scaled dot-product attention (PyTorch 2.0+ 自动使用 Flash Attention)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)

        return self.proj(attn_out)


class MLP(nn.Module):
    """DINOv2 MLP (GELU + fc1→fc2)"""

    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    """DINOv2 Transformer Block"""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = FusedQKVAttention(dim, num_heads)
        self.ls1 = LayerScale(dim)          # blocks.{i}.ls1.gamma
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)
        self.ls2 = LayerScale(dim)          # blocks.{i}.ls2.gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DINOv2ViT(nn.Module):
    """
    纯 PyTorch 实现的 DINOv2 ViT
    权重 key 格式与官方 .pth 完全对齐
    """

    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        patch_size: int = 14,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # Patch embedding (Conv2d)
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # CLS token + 位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # pos_embed: 1370 = 1369 patches (37*37, 对应 518x518) + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, 1370, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # 最终 LayerNorm
        self.norm = nn.LayerNorm(embed_dim)

    def _interpolate_pos_embed(self, num_patches: int) -> torch.Tensor:
        """插值位置编码以适配任意输入尺寸"""
        # self.pos_embed: [1, 1370, D]
        # 1370 = 1 (可能的 padding/extra) + 1369 (37*37)
        pos_len = self.pos_embed.shape[1]
        if num_patches + 1 == pos_len:
            return self.pos_embed

        # 分离第一个位置（通常对应 CLS 或 padding）
        extra = self.pos_embed[:, :1]
        spatial = self.pos_embed[:, 1:]

        N_old = spatial.shape[1]
        H_old = W_old = int(math.sqrt(N_old))
        H_new = W_new = int(math.sqrt(num_patches))

        if H_old * W_old != N_old:
            # 不是正方形 grid，用最近尺寸
            H_old = int(math.sqrt(N_old))
            W_old = N_old // H_old

        spatial = spatial.reshape(1, H_old, W_old, -1).permute(0, 3, 1, 2)
        spatial = F.interpolate(
            spatial, size=(H_new, W_new), mode="bilinear", align_corners=False
        )
        spatial = spatial.permute(0, 2, 3, 1).reshape(1, -1, self.embed_dim)

        return torch.cat([extra, spatial], dim=1)

    def forward(
        self, x: torch.Tensor, n_last: int = 4
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x:      [B, C, H, W]
            n_last: 返回最后 n 层的输出

        Returns:
            cls_token: [B, D]
            intermediates: List[(patch_out [B, N, D], cls_out [B, D])]
        """
        B, C, H, W = x.shape
        n_h = H // self.patch_size
        n_w = W // self.patch_size
        N = n_h * n_w

        # Patch embedding
        x = self.patch_embed.proj(x)            # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)         # [B, N, D]

        # 位置编码
        pos_embed = self._interpolate_pos_embed(N)
        # pos_embed 第一个位置给 CLS，其余给 patches
        x = x + pos_embed[:, 1:1 + N]            # [B, N, D]

        # 拼接 CLS token
        cls = self.cls_token.expand(B, -1, -1)   # [B, 1, D]
        cls = cls + pos_embed[:, :1]              # 给 CLS 也加位置编码
        x = torch.cat([cls, x], dim=1)            # [B, 1+N, D]

        # Transformer blocks
        depth = len(self.blocks)
        start_collect = depth - n_last
        intermediates = []

        for i, block in enumerate(self.blocks):
            x = block(x)
            if i >= start_collect:
                cls_out = x[:, 0]       # [B, D]
                patch_out = x[:, 1:]    # [B, N, D]
                intermediates.append((patch_out, cls_out))

        # 最终 norm
        x = self.norm(x)
        cls_final = x[:, 0]
        patch_final = x[:, 1:]
        # 替换最后一层为 norm 后的结果
        intermediates[-1] = (patch_final, cls_final)

        return cls_final, intermediates


# ─────────────────────────────────────────────
# 权重 URL 映射
# ─────────────────────────────────────────────

DINOV2_URLS = {
    "dinov2_vits14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
    "dinov2_vitb14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
    "dinov2_vitl14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
    "dinov2_vitg14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth",
}

DINOV2_CONFIGS = {
    "dinov2_vits14": {"embed_dim": 384, "depth": 12, "num_heads": 6},
    "dinov2_vitb14": {"embed_dim": 768, "depth": 12, "num_heads": 12},
    "dinov2_vitl14": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
    "dinov2_vitg14": {"embed_dim": 1536, "depth": 40, "num_heads": 24},
}


# ─────────────────────────────────────────────
# 特征提取器封装
# ─────────────────────────────────────────────

class DINOv2Extractor(nn.Module):
    """
    冻结的 DINOv2 特征提取器
    提取多层 patch-level 特征用于下游 INP-Former
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        out_indices: Optional[List[int]] = None,
        patch_size: int = 14,
        frozen: bool = True,
        weights_path: Optional[str] = None,
        torch_home: Optional[str] = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.out_indices = out_indices or [3, 6, 9, 11]

        if torch_home:
            os.environ["TORCH_HOME"] = torch_home

        self.backbone = self._load_backbone(model_name, weights_path)
        self.embed_dim = self.backbone.embed_dim

        if frozen:
            self._freeze()

    def _load_backbone(self, model_name: str, weights_path: Optional[str]) -> nn.Module:
        # ── 优先本地权重 ──
        if weights_path and os.path.exists(weights_path):
            print(f"[DINOv2] 从本地路径加载权重: {weights_path}")
            return self._load_from_local(model_name, weights_path)

        # ── torch.hub ──
        try:
            print(f"[DINOv2] 从 torch.hub 加载: {model_name}")
            return torch.hub.load(
                "facebookresearch/dinov2", model_name,
                pretrained=True, trust_repo=True,
            )
        except Exception as e:
            print(f"[DINOv2] torch.hub 失败: {e}")
            print(f"[DINOv2] 请手动下载: curl -L -o {model_name}.pth {DINOV2_URLS.get(model_name, 'N/A')}")
            raise

    def _load_from_local(self, model_name: str, weights_path: str) -> nn.Module:
        cfg = DINOV2_CONFIGS.get(model_name)
        if cfg is None:
            raise ValueError(f"未知模型: {model_name}")

        backbone = DINOv2ViT(
            embed_dim=cfg["embed_dim"],
            depth=cfg["depth"],
            num_heads=cfg["num_heads"],
            patch_size=self.patch_size,
        )

        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        # 去除 mask_token（推理不需要）
        state_dict.pop("mask_token", None)

        missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
        # 过滤掉无关的 missing（如 mask_token）
        missing = [k for k in missing if "mask_token" not in k]
        if missing:
            print(f"[DINOv2] 警告 - 缺少 {len(missing)} 个 key: {missing[:5]}...")
        if unexpected:
            print(f"[DINOv2] 警告 - 多余 {len(unexpected)} 个 key: {unexpected[:5]}...")

        print(f"[DINOv2] 权重加载成功 ({model_name}, {cfg['embed_dim']}d, {cfg['depth']}层)")
        return backbone

    def _freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    @torch.no_grad()
    def get_intermediate_layers(
        self, x: torch.Tensor, n_last_blocks: int = 4
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        cls_token, intermediates = self.backbone(x, n_last=n_last_blocks)
        patch_features = [patch for patch, _ in intermediates]
        cls_tokens = [cls for _, cls in intermediates]
        return cls_token, patch_features

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, H, W = x.shape
        n_h = H // self.patch_size
        n_w = W // self.patch_size

        cls_token, patch_features_list = self.get_intermediate_layers(
            x, n_last_blocks=len(self.out_indices)
        )

        return {
            "cls_token": cls_token,
            "patch_features": patch_features_list[-1],
            "multi_scale_features": list(patch_features_list),
            "num_patches_h": n_h,
            "num_patches_w": n_w,
        }

    @torch.no_grad()
    def extract_multi_view(self, views: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, V, C, H, W = views.shape
        views_flat = views.reshape(B * V, C, H, W)
        out = self.forward(views_flat)

        cls_tokens = out["cls_token"].reshape(B, V, -1)
        patch_features = out["patch_features"].reshape(B, V, -1, self.embed_dim)
        multi_scale = [
            f.reshape(B, V, -1, self.embed_dim)
            for f in out["multi_scale_features"]
        ]

        return {
            "cls_tokens": cls_tokens,
            "patch_features": patch_features,
            "multi_scale_features": multi_scale,
            "num_patches_h": out["num_patches_h"],
            "num_patches_w": out["num_patches_w"],
        }
