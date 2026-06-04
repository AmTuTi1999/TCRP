"""
Phase T* · Adversarial Concept Purity — GRL components and adversarial wrapper.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tcrp.model.tcrp_forecaster.forecaster import TCRPForecaster, TCRPOutput


# ---------------------------------------------------------------------------
# T*-01  Gradient Reversal Layer
# ---------------------------------------------------------------------------

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, alpha: float) -> Tensor:
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return -ctx.alpha * grad_output, None


class GRLLayer(nn.Module):
    """
    Gradient Reversal Layer (Ganin et al., 2016).

    Forward: identity.
    Backward (alignment path only): negates gradient by factor alpha.
    Forecast loss gradient is NOT reversed — that uses a separate forward path.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        return GradientReversal.apply(x, self.alpha)


# ---------------------------------------------------------------------------
# T*-02  Alpha schedule
# ---------------------------------------------------------------------------

def grl_alpha_schedule(
    epoch: int,
    max_epochs: int,
    warmup_epochs: int = 20,
    alpha_max: float = 1.0,
) -> float:
    """
    Ramps alpha from 0 to alpha_max over training.

    Phase 1 (0..warmup_epochs-1): alpha = 0 — cooperative training.
    Phase 2 (warmup_epochs..max_epochs): DANN sigmoid ramp.
    """
    if epoch < warmup_epochs:
        return 0.0
    p = (epoch - warmup_epochs) / max(max_epochs - warmup_epochs, 1)
    return alpha_max * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


# ---------------------------------------------------------------------------
# T*-03  Adversarial TCRP model wrapper
# ---------------------------------------------------------------------------

class AdversarialTCRPForecaster(nn.Module):
    """
    Wraps TCRPForecaster with a GRL on the alignment backward path.

    Exposes the same forward-pass outputs as TCRPForecaster so all downstream
    analysis (T-17 TCRP analysis pass, T-29–T-35 diagnostics) can be applied
    without modification.

    forward() returns (TCRPOutput, A_align):
    - TCRPOutput: standard output via the forecast path (no GRL)
    - A_align:    concept activations computed through the GRL for L_align

    The caller uses these for separate backward passes:
        L_forecast.backward(retain_graph=True)   # normal gradient to encoder
        L_align.backward()                       # reversed gradient to encoder
    """

    def __init__(self, base_model: TCRPForecaster, alpha: float = 0.0):
        super().__init__()
        self.base = base_model
        self.grl = GRLLayer(alpha=alpha)

    def set_alpha(self, alpha: float) -> None:
        self.grl.alpha = alpha

    def forward(self, x: Tensor) -> tuple[TCRPOutput, Tensor]:
        if x.dim() != 2:
            raise ValueError(f"Expected input shape (B, T), got {tuple(x.shape)}")

        B, T = x.shape
        segs = self.base.segmenter(x)          # (B, N, L)
        N = segs.shape[1]
        z = self.base.encoder(segs)             # (B, N, d)

        # --- Forecast path (no GRL) ---
        A_f = self.base.projection(z)           # (B, N, K)
        h, eta = self.base.pool(A_f)

        if self.base.config.probabilistic:
            mu, sigma = self.base.decoder(h)
            y_hat = mu
            self.base.last_sigma = sigma
        else:
            y_hat = self.base.decoder(h)
            self.base.last_sigma = None

        # --- Analytic concept scores (detached — no gradient) ---
        with torch.no_grad():
            flat_segs = segs.reshape(-1, segs.shape[-1])
            C_flat = self.base.scorer(flat_segs)
            C = C_flat.view(B, N, self.base.config.K)

        # --- Alignment path (through GRL — gradient reversed on backward) ---
        z_grl = self.grl(z)                    # identity forward, -alpha on backward
        A_align = self.base.projection(z_grl)  # (B, N, K)

        forecast_output = TCRPOutput(y_hat=y_hat, h=h, A=A_f, C=C, eta=eta)
        return forecast_output, A_align
