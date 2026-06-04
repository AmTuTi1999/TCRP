"""T-01 · Soft Monotonicity Score.

Paper: Def. 1, Eq. 1–2

Computes a differentiable measure of monotonic increase/decrease tendency.
"""

from typing import NamedTuple

import torch
from torch import Tensor


class MonotonicityScores(NamedTuple):
    """Container for soft monotonicity scores."""

    mu: Tensor  # sigmoid of avg. first differences, shape ()
    mu_signed: Tensor  # range (-1, 1), shape ()
    mu_mag: Tensor  # absolute signed monotonicity, range (0, 1), shape ()


def soft_monotonicity(s: Tensor, alpha: float = 5.0) -> MonotonicityScores:
    """Compute soft monotonicity score for a sequence.

    Args:
        s: Input sequence of shape (L,) or (B, L)
        alpha: Temperature parameter controlling sigmoid sharpness

    Returns:
        MonotonicityScores named tuple containing:
        - mu: sigmoid of average first differences, shape ()
        - mu_signed: signed monotonicity in range (-1, 1), shape ()
        - mu_mag: magnitude of signed monotonicity in range (0, 1), shape ()

    Interpretation:
        - mu_signed ≈ 1: strictly increasing sequence
        - mu_signed ≈ -1: strictly decreasing sequence
        - mu_signed ≈ 0: no clear monotonic trend (random walk)
    """
    # Handle both single sequence (L,) and batched (B, L)
    if s.dim() == 1:
        s = s.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    delta = s[:, 1:] - s[:, :-1]  # shape (B, L-1)
    mu = torch.sigmoid(alpha * delta).mean(dim=1)  # shape (B,)
    mu_signed = 2 * mu - 1  # shape (B,)
    mu_mag = mu_signed.abs()  # shape (B,)

    if squeeze_output:
        mu = mu.squeeze(0)
        mu_signed = mu_signed.squeeze(0)
        mu_mag = mu_mag.squeeze(0)

    return MonotonicityScores(mu=mu, mu_signed=mu_signed, mu_mag=mu_mag)
