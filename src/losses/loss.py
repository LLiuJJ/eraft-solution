"""
损失函数模块
用于 INP-Former 的训练
"""
import torch
import torch.nn as nn


class INPFormerLoss(nn.Module):
    """
    INP-Former 综合损失函数

    包含三个部分：
    1. NLL CLS 损失：CLS token 的负对数似然，衡量全局异常
    2. NLL View 损失：视角 token 的负对数似然，衡量视角一致性
    3. Flow 正则化：防止 Flow 退化，保持可逆变换的数值稳定性
    """

    def __init__(
        self,
        lambda_cls: float = 1.0,
        lambda_view: float = 0.5,
        lambda_reg: float = 0.01,
        lambda_log_det: float = 0.01,
    ):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_view = lambda_view
        self.lambda_reg = lambda_reg
        self.lambda_log_det = lambda_log_det

    def forward(self, outputs: dict) -> dict:
        """
        Args:
            outputs: INPFormer.forward() 的输出

        Returns:
            dict: 各项损失和总损失
        """
        nll_cls = outputs["nll_cls"].mean()     # [B] -> scalar
        nll_view = outputs["nll_view"].mean()   # [B, V] -> scalar

        # Flow 正则化：对隐变量 z 施加单位高斯先验
        z_cls = outputs["z_cls"]
        z_view = outputs["z_view"]

        # z 应该接近标准正态分布
        reg_cls = (z_cls ** 2).mean()
        reg_view = (z_view ** 2).mean()
        reg_loss = reg_cls + reg_view

        # log_det 正则化：惩罚过大的雅可比行列式，防止 Flow "体积膨胀" 作弊
        log_det_reg = torch.tensor(0.0, device=nll_cls.device)
        if "log_det_cls" in outputs:
            log_det_reg = log_det_reg + (outputs["log_det_cls"] ** 2).mean()
        if "log_det_view" in outputs:
            log_det_reg = log_det_reg + (outputs["log_det_view"] ** 2).mean()

        total_loss = (
            self.lambda_cls * nll_cls
            + self.lambda_view * nll_view
            + self.lambda_reg * reg_loss
            + self.lambda_log_det * log_det_reg
        )

        return {
            "total_loss": total_loss,
            "nll_cls": nll_cls.detach(),
            "nll_view": nll_view.detach(),
            "reg_loss": reg_loss.detach(),
            "log_det_reg": log_det_reg.detach(),
        }
