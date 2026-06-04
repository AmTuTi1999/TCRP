"""Loss utilities for TCRP training."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from tcrp.model.tcrp_forecaster.components.bottleneck import alignment_loss, ConceptProjection


@dataclass
class LossBundle:
    forecast_loss: Tensor
    align_loss: Tensor
    reg_loss: Tensor
    total_loss: Tensor
    stab_loss: Optional[Tensor] = None


class TCRPLoss:
    def __init__(self, lambda1: float, lambda2: float):
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)

    def __call__(self, y_hat: Tensor, y_true: Tensor, A: Tensor, C: Tensor, projection: ConceptProjection) -> LossBundle:
        forecast_loss = F.mse_loss(y_hat, y_true)
        align_loss_val = alignment_loss(A, C)
        reg_loss_val = torch.norm(projection.linear.weight, p='fro')
        total_loss = forecast_loss + self.lambda1 * align_loss_val + self.lambda2 * reg_loss_val
        return LossBundle(
            forecast_loss=forecast_loss,
            align_loss=align_loss_val,
            reg_loss=reg_loss_val,
            total_loss=total_loss,
        )
