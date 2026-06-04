"""T-03e · Structural Break Concepts.

Paper: Def. 7, Eqs. 20–22

Three differentiable break detectors comparing the first and second halves of a segment:
  - b_mu:       mean-level break, range [0, 1)
  - b_sigma:    volatility-regime break, range [0, 1)
  - b_mu_tilde: slope-direction break, range (-1, 1)

All outputs are differentiable w.r.t. the input.
"""

from typing import TypedDict

import torch
from torch import Tensor

from .monotonicity import soft_monotonicity


class BreakScores(TypedDict):
    """TypedDict holding structural break concept scores."""

    b_mu: Tensor  # scalar or (B,), range [0, 1)
    b_sigma: Tensor  # scalar or (B,), range [0, 1)
    b_mu_tilde: Tensor  # scalar or (B,), range (-1, 1)


def break_scores(
    s: Tensor,
    alpha: float = 5.0,
    eps: float = 1e-8,
) -> BreakScores:
    """Compute structural break concept scores for a time series segment.

    Splits the segment at the midpoint and compares distributional properties
    of the two halves to detect level shifts, volatility regime changes, and
    slope reversals.

    Args:
        s:     Input sequence, shape (L,) or (B, L).  L must be >= 4.
        alpha: Temperature for soft_monotonicity applied to each half.
        eps:   Numerical stability constant.

    Returns:
        BreakScores dict with keys:
          b_mu       — tanh-normalised mean-level shift, range [0, 1)
          b_sigma    — tanh-normalised log-volatility change, range [0, 1)
          b_mu_tilde — signed slope change between halves, range (-1, 1)
    """
    batched = s.dim() == 2
    if not batched:
        s = s.unsqueeze(0)  # (1, L)

    B, L = s.shape
    mid = L // 2
    s1, s2 = s[:, :mid], s[:, mid:]  # (B, mid), (B, L-mid)

    # Full-window increment std for mean-break normalisation
    delta = s[:, 1:] - s[:, :-1]  # (B, L-1)
    sigma_hat = delta.std(dim=-1, unbiased=False)  # (B,)

    # --- Mean break ---
    mean_diff = (s2.mean(dim=-1) - s1.mean(dim=-1)).abs()  # (B,)
    b_mu_raw = torch.tanh(mean_diff / (sigma_hat + eps))
    b_mu = torch.where(sigma_hat < 1e-8, torch.zeros_like(b_mu_raw), b_mu_raw)

    # --- Variance break ---
    delta1 = s1[:, 1:] - s1[:, :-1]  # (B, mid-1)
    delta2 = s2[:, 1:] - s2[:, :-1]  # (B, L-mid-1)
    sigma1 = delta1.std(dim=-1, unbiased=False)  # (B,)
    sigma2 = delta2.std(dim=-1, unbiased=False)  # (B,)
    log_ratio = (sigma2 / (sigma1 + eps)).abs().log().abs()  # |log(sigma2/sigma1)|
    b_sigma_raw = torch.tanh(log_ratio)
    degenerate = (sigma1 < 1e-8) | (sigma2 < 1e-8)
    b_sigma = torch.where(degenerate, torch.zeros_like(b_sigma_raw), b_sigma_raw)

    # --- Slope break ---
    mu1 = soft_monotonicity(s1, alpha=alpha).mu_signed  # (B,)
    mu2 = soft_monotonicity(s2, alpha=alpha).mu_signed  # (B,)
    b_mu_tilde = (mu2 - mu1) / 2.0  # (B,), range (-1, 1)

    if not batched:
        b_mu = b_mu.squeeze(0)
        b_sigma = b_sigma.squeeze(0)
        b_mu_tilde = b_mu_tilde.squeeze(0)

    return BreakScores(b_mu=b_mu, b_sigma=b_sigma, b_mu_tilde=b_mu_tilde)
