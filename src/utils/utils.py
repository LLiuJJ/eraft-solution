"""
训练工具函数：日志、checkpoint、学习率调度
"""
import os
import time
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42):
    """设置随机种子，确保可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AverageMeter:
    """滑动平均计算器"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


class MetricTracker:
    """训练指标追踪器"""

    def __init__(self, keys: list):
        self.keys = keys
        self.meters = {k: AverageMeter() for k in keys}

    def reset(self):
        for m in self.meters.values():
            m.reset()

    def update(self, key: str, val: float, n: int = 1):
        self.meters[key].update(val, n)

    def avg_all(self) -> dict:
        return {k: m.avg for k, m in self.meters.items()}


def get_lr_scheduler(optimizer, cfg):
    """构建学习率调度器"""
    if cfg.train.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.train.epochs,
            eta_min=1e-6,
        )
    elif cfg.train.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.train.epochs // 3,
            gamma=0.1,
        )
    else:
        scheduler = None
    return scheduler


def save_checkpoint(state: dict, path: str):
    """保存模型 checkpoint"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"[Checkpoint] 已保存至 {path}")


def load_checkpoint(path: str, model, optimizer=None, device="cpu"):
    """加载模型 checkpoint"""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    epoch = ckpt.get("epoch", 0)
    print(f"[Checkpoint] 已加载 {path}, epoch={epoch}")
    return epoch
