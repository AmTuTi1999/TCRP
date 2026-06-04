"""T-03f · Distributional Shape Concepts.

Paper: Def. 8, Eqs. 23–25

Three differentiable descriptors of the increment distribution for a segment:
  - varsigma: signed skewness of increments, clamped to [-3, 3]
  - kappa4:   signed excess kurtosis of increments, clamped to [-3, 10]
  - j:        soft jump indicator, range (0, 1) via sigmoid

All outputs are differentiable w.r.t. the input.
"""

from typing import TypedDict

import torch
from torch import Tensor


class ShapeScores(TypedDict):
    """TypedDict holding distributional shape concept scores."""

    varsigma: Tensor  # scalar or (B,), range [-3, 3]
    kappa4: Tensor  # scalar or (B,), range [-3, 10]
    j: Tensor  # scalar or (B,), range (0, 1)


def shape_scores(
    s: Tensor,
    gamma_j: float = 2.0,
    jump_threshold: float = 3.0,
    eps: float = 1e-8,
) -> ShapeScores:
    """Compute distributional shape concept scores for a time series segment.

    Args:
        s:               Input sequence, shape (L,) or (B, L).
        gamma_j:         Sigmoid sharpness for the soft jump indicator.
        jump_threshold:  Number of std devs above which a max increment is a jump.
        eps:             Numerical stability constant.

    Returns:
        ShapeScores dict with keys:
          varsigma — signed skewness of first differences, clamped to [-3, 3]
          kappa4   — signed excess kurtosis of first differences, clamped to [-3, 10]
          j        — soft jump indicator in (0, 1); > 0.5 when max|delta| > threshold * std(delta)
    """
    batched = s.dim() == 2
    if not batched:
        s = s.unsqueeze(0)  # (1, L)

    delta = s[:, 1:] - s[:, :-1]  # (B, L-1)
    delta_c = delta - delta.mean(dim=-1, keepdim=True)  # centred, (B, L-1)

    std_delta = delta.std(dim=-1, unbiased=False)  # (B,)
    std3 = std_delta.pow(3) + eps
    std4 = std_delta.pow(4) + eps

    varsigma = delta_c.pow(3).mean(dim=-1) / std3  # (B,)
    varsigma = varsigma.clamp(-3.0, 3.0)

    kappa4 = delta_c.pow(4).mean(dim=-1) / std4 - 3.0  # (B,)
    kappa4 = kappa4.clamp(-3.0, 10.0)

    max_abs_delta = (
        delta.abs().max(dim=-1).values
    )  # (B,), differentiable via smooth max in bwd
    j = torch.sigmoid(gamma_j * (max_abs_delta - jump_threshold * std_delta))  # (B,)

    if not batched:
        varsigma = varsigma.squeeze(0)
        kappa4 = kappa4.squeeze(0)
        j = j.squeeze(0)

    return ShapeScores(varsigma=varsigma, kappa4=kappa4, j=j)
