"""T-03c · Volatility Concepts.

Paper: Def. 5, Eqs. 14–16

Computes three differentiable volatility descriptors for a time series segment:
  - sigma_tilde: realised volatility normalised by training-set std, range [0, ∞)
  - mu_v:        soft monotonicity of |delta| — direction of vol trend, range (-1, 1)
  - psi:         ARCH clustering — lag-1 ACF of squared increments, range (-1, 1)

All outputs are differentiable w.r.t. the input.
"""

from typing import TypedDict

import torch
from torch import Tensor

from .monotonicity import soft_monotonicity


class VolatilityScores(TypedDict):
    """TypedDict holding the three volatility concept scores."""

    sigma_tilde: Tensor  # scalar or (B,), range [0, ∞)
    mu_v: Tensor  # scalar or (B,), range (-1, 1)
    psi: Tensor  # scalar or (B,), range (-1, 1)


def volatility_scores(
    s: Tensor,
    train_std: float = 1.0,
    alpha: float = 5.0,
    eps: float = 1e-8,
) -> VolatilityScores:
    """Compute volatility concept scores for a time series segment.

    Args:
        s:         Input sequence, shape (L,) or (B, L).
        train_std: Training-set standard deviation used to normalise sigma.
                   Pass dataset-level statistics computed on the training split only.
        alpha:     Temperature for soft_monotonicity applied to |delta|.
        eps:       Numerical stability constant.

    Returns:
        VolatilityScores dict with keys:
          sigma_tilde — realised vol / train_std, range [0, ∞)
          mu_v        — signed vol trend in (-1, 1): >0 rising, <0 falling
          psi         — ARCH clustering coefficient in (-1, 1)
    """
    batched = s.dim() == 2
    if not batched:
        s = s.unsqueeze(0)  # (1, L)

    delta = s[:, 1:] - s[:, :-1]  # (B, L-1)

    # --- Realised volatility ---
    sigma = delta.pow(2).mean(dim=-1).sqrt()  # (B,)
    sigma_tilde = sigma / (train_std + eps)  # (B,), dimensionless

    # --- Volatility trend ---
    v = delta.abs()  # (B, L-1)
    mono = soft_monotonicity(v, alpha=alpha)
    mu_v = mono.mu_signed  # (B,), range (-1, 1)

    # --- ARCH clustering (lag-1 ACF of squared increments) ---
    d2 = delta.pow(2)  # (B, L-1)
    x, y = d2[:, :-1], d2[:, 1:]  # (B, L-2) each
    x_c = x - x.mean(dim=-1, keepdim=True)
    y_c = y - y.mean(dim=-1, keepdim=True)
    cov = (x_c * y_c).mean(dim=-1)  # (B,)
    var_x = x_c.pow(2).mean(dim=-1)
    var_y = y_c.pow(2).mean(dim=-1)
    denom = (var_x * var_y).sqrt()  # (B,)
    # Degenerate: near-constant squared increments → psi = 0
    psi = torch.where(denom > 1e-8, cov / denom, torch.zeros_like(cov))

    if not batched:
        sigma_tilde = sigma_tilde.squeeze(0)
        mu_v = mu_v.squeeze(0)
        psi = psi.squeeze(0)

    return VolatilityScores(sigma_tilde=sigma_tilde, mu_v=mu_v, psi=psi)
