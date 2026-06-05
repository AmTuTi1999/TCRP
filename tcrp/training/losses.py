"""Loss utilities for TCRP training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from tcrp.model.tcrp_forecaster.components.bottleneck import (
    ConceptProjection,
    concept_magnitude_loss,
    stability_loss,
    weighted_alignment_loss,
)


@dataclass
class LossBundle:
    """Dataclass bundling all TCRP loss components for logging and optimisation."""

    forecast_loss: Tensor
    align_loss: Tensor
    mag_loss: Tensor
    stab_loss: Tensor
    reg_loss: Tensor
    total_loss: Tensor


class TCRPLoss:
    """Combined TCRP training loss.

    Three alignment strategies act together to keep the learned concept
    activations A grounded in the analytic concept scores C:

    lambda1  — weighted Pearson alignment: correlation penalty scaled by
               mean |C_k| so strongly-present concepts get more pull and
               absent concepts are silenced rather than skipped.

    lambda4  — magnitude alignment: penalises (mean|A_k| - mean|C_k|)^2
               per concept, forcing the projection to match the absolute
               scale of each analytic score.

    lambda3  — stability: MSE on consecutive-segment deltas of A vs C,
               encouraging smooth concept trajectories that track C over
               the window.

    lambda2  — Frobenius regularisation on the projection weight matrix.
    """

    def __init__(
        self,
        lambda1: float,
        lambda2: float,
        lambda3: float = 0.0,
        lambda4: float = 0.0,
    ):
        """Initialise with per-term loss weights."""
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.lambda3 = float(lambda3)
        self.lambda4 = float(lambda4)

    def __call__(
        self,
        y_hat: Tensor,
        y_true: Tensor,
        A: Tensor,
        C: Tensor,
        projection: ConceptProjection,
    ) -> LossBundle:
        """Compute and return a LossBundle."""
        forecast_loss = F.mse_loss(y_hat, y_true)
        align_loss_val = weighted_alignment_loss(A, C)
        mag_loss_val = concept_magnitude_loss(A, C)
        stab_loss_val = stability_loss(A, C)
        reg_loss_val = torch.norm(projection.linear.weight, p="fro")

        total_loss = (
            forecast_loss
            + self.lambda1 * align_loss_val
            + self.lambda4 * mag_loss_val
            + self.lambda3 * stab_loss_val
            + self.lambda2 * reg_loss_val
        )
        return LossBundle(
            forecast_loss=forecast_loss,
            align_loss=align_loss_val,
            mag_loss=mag_loss_val,
            stab_loss=stab_loss_val,
            reg_loss=reg_loss_val,
            total_loss=total_loss,
        )
