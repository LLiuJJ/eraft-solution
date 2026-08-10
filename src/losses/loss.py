"""
损失函数模块
用于 INP-Former 的训练

训练信号: 0.5 * ||z||² (z 空间到原点的距离)
- 正常样本: Flow 应将 x 映射到 z ≈ 0 附近 (||z||² 小)
- 异常样本 (推理时): ||z||² 大 → 异常得分高
- 去掉了 log_det，防止 Flow 通过膨胀雅可比行列式作弊
"""
import torch
import torch.nn as nn


class INPFormerLoss(nn.Module):
    """
    INP-Former 损失函数

    核心: 最小化 z 空间到原点的距离
    - nll_cls:   0.5 * ||z_cls||²  (全局特征)
    - nll_view:  0.5 * ||z_view||² (视角特征)

    辅助: z 方差正则化 (推动 z 各维度方差接近 1)
    """

    def __init__(
        self,
        lambda_cls: float = 1.0,
        lambda_view: float = 0.5,
        lambda_var: float = 1.0,      # 0.1→1.0: 大幅增强反塌缩力度
        lambda_margin: float = 0.5,   # 新增: 强制 z 均值范数下界
    ):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_view = lambda_view
        self.lambda_var = lambda_var
        self.lambda_margin = lambda_margin

    def forward(self, outputs: dict) -> dict:
        """
        Args:
            outputs: INPFormer.forward() 的输出

        Returns:
            dict: 各项损失和总损失
        """
        nll_cls = outputs["nll_cls"].mean()     # [B] -> scalar
        nll_view = outputs["nll_view"].mean()   # [B, V] -> scalar

        # z 方差正则化: 推动 z 各维度方差接近 1 (标准高斯)
        z_cls = outputs["z_cls"]          # [B, D]
        z_view = outputs["z_view"]        # [B, V, D]

        var_cls = z_cls.var(dim=0)        # [D]
        var_view = z_view.reshape(-1, z_view.size(-1)).var(dim=0)  # [D]
        # 方差偏离 1 的惩罚
        var_reg = ((var_cls - 1.0) ** 2).mean() + ((var_view - 1.0) ** 2).mean()

        # 反塌缩 margin 项: 强制 batch 内 z 均值范数不低于 margin
        # 如果 z 均值范数 < margin, 给予惩罚 (hinge loss)
        cls_norm = z_cls.norm(dim=-1).mean()     # 标量
        view_norm = z_view.norm(dim=-1).mean()   # 标量
        margin_reg = torch.relu(1.0 - cls_norm).mean() + torch.relu(1.0 - view_norm).mean()

        total_loss = (
            self.lambda_cls * nll_cls
            + self.lambda_view * nll_view
            + self.lambda_var * var_reg
            + self.lambda_margin * margin_reg
        )

        return {
            "total_loss": total_loss,
            "nll_cls": nll_cls.detach(),
            "nll_view": nll_view.detach(),
            "reg_loss": var_reg.detach(),
            "margin_loss": margin_reg.detach(),
            "log_det_reg": torch.tensor(0.0),
        }
