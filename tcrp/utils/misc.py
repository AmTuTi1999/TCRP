"""Miscellaneous project-wide utilities."""
from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + all GPUs) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def elapsed_str(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string.

    Examples: 45 → '45s', 90 → '1m 30s', 3725 → '1h 02m 05s'
    """
    s = int(seconds)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"
