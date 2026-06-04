"""Loss utilities for TCRP training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from tcrp.model.tcrp_forecaster.components.bottleneck import (
    ConceptProjection,
    alignment_loss,
)


@dataclass
class LossBundle:
    """Dataclass bundling all TCRP loss components for logging and optimisation."""

    forecast_loss: Tensor
    align_loss: Tensor
    reg_loss: Tensor
    total_loss: Tensor
    stab_loss: Tensor | None = None


class TCRPLoss:
    """Computes the combined TCRP training loss from forecast, alignment, and regularisation terms."""

    def __init__(self, lambda1: float, lambda2: float):
        """Initialize TCRPLoss with alignment and regularisation loss weights."""
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)

    def __call__(
        self,
        y_hat: Tensor,
        y_true: Tensor,
        A: Tensor,
        C: Tensor,
        projection: ConceptProjection,
    ) -> LossBundle:
        """Compute and return a LossBundle from predictions, targets, and concept activations."""
        forecast_loss = F.mse_loss(y_hat, y_true)
        align_loss_val = alignment_loss(A, C)
        reg_loss_val = torch.norm(projection.linear.weight, p="fro")
        total_loss = (
            forecast_loss + self.lambda1 * align_loss_val + self.lambda2 * reg_loss_val
        )
        return LossBundle(
            forecast_loss=forecast_loss,
            align_loss=align_loss_val,
            reg_loss=reg_loss_val,
            total_loss=total_loss,
        )
