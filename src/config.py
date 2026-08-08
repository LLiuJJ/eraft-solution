"""
DINOv2 + INP-Former 异常检测配置文件
Real-IAD Variety 数据集
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    """数据相关配置"""
    root_dir: str = "data"
    train_dir: str = "Train"
    test_dir: str = "Test_A"
    num_views: int = 5           # 每个样本的视角数
    image_size: int = 518        # DINOv2 ViT-B/14 最佳输入尺寸 (14*37)
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    num_workers: int = 4
    batch_size: int = 8          # 每个 batch 的样本数（多视角）
    pin_memory: bool = True


@dataclass
class DINOv2Config:
    """DINOv2 特征提取器配置"""
    model_name: str = "dinov2_vitb14"   # ViT-B/14
    patch_size: int = 14
    embed_dim: int = 768
    frozen: bool = True                  # 冻结权重
    # 提取多层特征
    out_indices: List[int] = field(default_factory=lambda: [3, 6, 9, 11])
    use_cls_token: bool = True
    # GPU 服务器适配
    weights_path: Optional[str] = None   # 本地权重路径（离线环境，留 None 则在线下载）
    torch_home: Optional[str] = None     # torch hub 缓存目录（如 /data/cache/torch_home）


@dataclass
class INPFormerConfig:
    """INP-Former 模型配置"""
    # Transformer 编码器
    d_model: int = 256                   # 特征嵌入维度
    n_heads: int = 8                     # 注意力头数
    n_layers: int = 4                    # Transformer 层数
    dim_ff: int = 1024                   # FFN 中间层维度
    dropout: float = 0.1
    # 可逆神经网络过程 (Normalizing Flow)
    n_flow_layers: int = 6              # 流层数
    flow_hidden_dim: int = 256          # 流隐藏层维度
    coupling_hidden: int = 128          # Coupling layer 隐藏层
    # 多视角融合
    view_fusion: str = "cross_attention"  # cross_attention | mean | concat
    # 异常得分
    score_type: str = "nll"              # nll (负对数似然) | mahalanobis


@dataclass
class TrainConfig:
    """训练配置"""
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-5
    scheduler: str = "cosine"            # cosine | step
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    log_interval: int = 10
    save_interval: int = 10
    eval_interval: int = 5
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    seed: int = 42
    device: str = "cuda"                 # cuda | cpu | mps
    # GPU 训练加速
    amp: bool = True                     # 启用混合精度训练 (FP16)
    cudnn_benchmark: bool = True         # cuDNN 自动寻优（固定输入尺寸时开启）


@dataclass
class Config:
    """总配置"""
    data: DataConfig = field(default_factory=DataConfig)
    dinov2: DINOv2Config = field(default_factory=DINOv2Config)
    inpformer: INPFormerConfig = field(default_factory=INPFormerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    categories: Optional[List[str]] = None  # None 表示使用全部类别


def get_config(category: Optional[str] = None) -> Config:
    """获取配置，可指定单一类别训练"""
    cfg = Config()
    if category is not None:
        cfg.categories = [category]
    # 自动检测设备
    import torch
    if torch.cuda.is_available():
        cfg.train.device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        cfg.train.device = "mps"
        cfg.train.amp = False            # MPS 不支持 AMP
    else:
        cfg.train.device = "cpu"
        cfg.train.amp = False
        cfg.data.pin_memory = False
    return cfg
