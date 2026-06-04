"""GBM Stochastic Concepts.

Estimates Geometric Brownian Motion parameters (drift μ, volatility σ) per segment
to capture stochastic dynamics. High volatility indicates segments dominated by noise
rather than deterministic temporal structure.

GBM model: dS = μS dt + σS dW
MLE from log-returns: σ̂ = std(r), μ̂ = mean(r) + σ̂²/2  (Itô's lemma correction)
"""

from typing import NamedTuple

import torch
from torch import Tensor


class GBMScores(NamedTuple):
    """Named tuple holding GBM drift and volatility estimates."""

    gbm_drift: Tensor  # normalized drift rate, range (-1, 1)
    gbm_vol: Tensor  # normalized volatility, range [0, 1)


def gbm_scores(
    s: Tensor,
    drift_scale: float = 5.0,
    vol_scale: float = 5.0,
    eps: float = 1e-8,
) -> GBMScores:
    """Estimate GBM drift and volatility from a time series segment.

    The segment is shifted to be strictly positive before computing log-returns,
    making this compatible with z-scored inputs that may contain negative values.
    Both outputs are fully differentiable w.r.t. the input.

    Args:
        s: Input sequence of shape (L,) or (B, L)
        drift_scale: Sensitivity scale for tanh normalization of μ̂
        vol_scale: Sensitivity scale for exponential normalization of σ̂
        eps: Small constant for numerical stability inside log

    Returns:
        GBMScores with:
        - gbm_drift: tanh(drift_scale * μ̂) ∈ (-1, 1)
        - gbm_vol:   1 - exp(-vol_scale * σ̂) ∈ [0, 1)
    """
    if s.dim() == 1:
        s = s.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    # Shift to strictly positive so log-returns are well-defined
    s_min = s.min(dim=1, keepdim=True).values
    s_pos = s - s_min + 1.0  # (B, L), all values >= 1.0

    # Log-returns: r[b, l] = log(s_pos[b, l+1] / s_pos[b, l])
    log_s = torch.log(s_pos + eps)  # (B, L)
    r = log_s[:, 1:] - log_s[:, :-1]  # (B, L-1)

    # MLE estimates: unbiased std, drift corrected via Itô's lemma
    sigma_hat = r.std(dim=1, unbiased=True)  # (B,), >= 0
    mu_hat = r.mean(dim=1) + 0.5 * sigma_hat.pow(2)  # (B,)

    gbm_drift = torch.tanh(drift_scale * mu_hat)  # (-1, 1)
    gbm_vol = 1.0 - torch.exp(-vol_scale * sigma_hat)  # [0, 1)

    if squeeze_output:
        gbm_drift = gbm_drift.squeeze(0)
        gbm_vol = gbm_vol.squeeze(0)

    return GBMScores(gbm_drift=gbm_drift, gbm_vol=gbm_vol)
