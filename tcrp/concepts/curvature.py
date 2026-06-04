"""T-02 · Soft Curvature Score and Observed Tendency.

Paper: Def. 2, Eq. 3–4

Computes curvature (concavity/convexity) and observed tendency (interaction of
monotonicity and curvature).
"""

from typing import NamedTuple

import torch
from torch import Tensor


class CurvatureScores(NamedTuple):
    """Container for soft curvature scores."""

    kappa_signed: Tensor  # signed curvature, range (-1, 1), shape () or (B,)
    tau: Tensor  # observed tendency (mu_signed * kappa_signed), range (-1, 1), shape () or (B,)


def soft_curvature(s: Tensor, mu_signed: Tensor, beta: float = 5.0) -> CurvatureScores:
    """Compute soft curvature score for a sequence.

    Args:
        s: Input sequence of shape (L,) or (B, L)
        mu_signed: Pre-computed signed monotonicity from soft_monotonicity(),
                   shape () or (B,)
        beta: Temperature parameter controlling sigmoid sharpness

    Returns:
        CurvatureScores named tuple containing:
        - kappa_signed: signed curvature in range (-1, 1)
        - tau: observed tendency (mu_signed * kappa_signed) in range (-1, 1)

    Interpretation (4-regime table):
        - mu_signed > 0, kappa_signed > 0: accelerating rise (tau > 0)
        - mu_signed > 0, kappa_signed < 0: decelerating rise (tau < 0)
        - mu_signed < 0, kappa_signed < 0: accelerating fall (tau > 0)
        - mu_signed < 0, kappa_signed > 0: decelerating fall (tau < 0)
    """
    if s.dim() == 1:
        s = s.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    delta = s[:, 1:] - s[:, :-1]  # shape (B, L-1)
    delta2 = delta[:, 1:] - delta[:, :-1]  # shape (B, L-2)
    kappa = torch.sigmoid(beta * delta2).mean(dim=1)  # shape (B,)
    kappa_signed = 2 * kappa - 1  # shape (B,)
    if mu_signed.dim() == 0:
        mu_signed = mu_signed.unsqueeze(0)

    tau = mu_signed * kappa_signed  # shape (B,)

    if squeeze_output:
        kappa_signed = kappa_signed.squeeze(0)
        tau = tau.squeeze(0)

    return CurvatureScores(kappa_signed=kappa_signed, tau=tau)
