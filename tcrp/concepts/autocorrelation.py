"""T-03d · Autocorrelation Concepts.

Paper: Def. 6, Eqs. 17–19

Computes differentiable autocorrelation-based descriptors for a time series segment:
  - rho:      lag-k increment ACF (Pearson), shape (k_max,), range (-1, 1)
  - theta_hat: mean-reversion speed (OU estimator), clamped to [-2, 2]
  - z:         z-score of last observation relative to segment mean, clamped to [-3, 3]

All outputs are differentiable w.r.t. the input via standard Pearson / std ops.
"""

from typing import TypedDict

import torch
from torch import Tensor


class AutocorrelationScores(TypedDict, total=False):
    """TypedDict holding autocorrelation concept scores."""

    rho: Tensor  # shape (k_max,), range (-1, 1)
    theta_hat: Tensor  # scalar, range [-2, 2]
    z: Tensor  # scalar, range [-3, 3]


def _pearson_corr(x: Tensor, y: Tensor, eps: float = 1e-8) -> Tensor:
    """Pearson correlation along last dim; x, y: (..., N)."""
    x = x - x.mean(dim=-1, keepdim=True)
    y = y - y.mean(dim=-1, keepdim=True)
    cov = (x * y).mean(dim=-1)
    denom = x.std(dim=-1, unbiased=False) * y.std(dim=-1, unbiased=False) + eps
    return cov / denom


def autocorrelation_scores(
    s: Tensor,
    k_max: int = 2,
    include_rho: bool = True,
    include_theta: bool = True,
    include_z: bool = True,
    eps: float = 1e-8,
) -> AutocorrelationScores:
    """Compute autocorrelation-based concept scores for a time series segment.

    Args:
        s:             Input sequence, shape (L,) or (B, L).
        k_max:         Maximum lag for the increment ACF.
        include_rho:   Include lag-k increment ACF tensor.
        include_theta: Include mean-reversion speed estimate.
        include_z:     Include z-score of the last observation.
        eps:           Numerical stability constant.

    Returns:
        Dict with a subset of keys {rho, theta_hat, z} depending on include_* flags.
        rho       — Tensor(k_max,) or (B, k_max): Pearson ACF of first differences at lags 1..k_max
        theta_hat — scalar or (B,): OU mean-reversion coefficient, clamped to [-2, 2]
        z         — scalar or (B,): standardised last observation, clamped to [-3, 3]
    """
    batched = s.dim() == 2
    if not batched:
        s = s.unsqueeze(0)  # (1, L)

    delta = s[:, 1:] - s[:, :-1]  # (B, L-1)
    result: AutocorrelationScores = {}

    if include_rho:
        rhos = []
        for k in range(1, k_max + 1):
            rhos.append(_pearson_corr(delta[:, :-k], delta[:, k:], eps=eps))  # (B,)
        rho = torch.stack(rhos, dim=-1)  # (B, k_max)
        result["rho"] = rho if batched else rho.squeeze(0)

    if include_theta:
        s_lag = s[:, :-1]  # OU process: delta_t = theta * s_{t-1} + noise
        corr_sd = _pearson_corr(s_lag, delta, eps=eps)  # (B,)
        std_delta = delta.std(dim=-1, unbiased=False)  # (B,)
        std_s_lag = s_lag.std(dim=-1, unbiased=False)  # (B,)
        theta_hat = -corr_sd * std_delta / (std_s_lag + eps)
        theta_hat = theta_hat.clamp(-2.0, 2.0)
        result["theta_hat"] = theta_hat if batched else theta_hat.squeeze(0)

    if include_z:
        z = (s[:, -1] - s.mean(dim=-1)) / (s.std(dim=-1, unbiased=False) + eps)
        z = z.clamp(-3.0, 3.0)
        result["z"] = z if batched else z.squeeze(0)

    return result
