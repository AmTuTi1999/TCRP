"""
T-04 · Full Temporal Concept Vector
Paper: Eq. 6, Table 2

Combines all analytic concept scores into a single differentiable vector.
Optionally extends with GBM stochastic concepts (drift, volatility) to
account for the stochastic component of each segment's dynamics.
"""

from typing import List
import torch
import torch.nn as nn
from torch import Tensor

from .monotonicity import soft_monotonicity
from .curvature import soft_curvature
from .periodicity import periodicity_score
from .stochastic import gbm_scores


class ConceptScorer(nn.Module):
    """
    Computes the full temporal concept vector for time series segments.

    This is a pure analysis module with no learned parameters.
    All computations are fully differentiable w.r.t. input.

    When include_gbm=True, two additional GBM stochastic concepts are appended:
      - gbm_drift ∈ (-1, 1): estimated log-price drift rate (direction + strength)
      - gbm_vol   ∈ [0, 1):  estimated volatility (0 = deterministic, 1 = fully noisy)
    K is increased by 2 accordingly.
    """

    def __init__(
        self,
        alpha: float = 5.0,
        beta: float = 5.0,
        periods: List[int] = None,
        include_gbm: bool = True,
        drift_scale: float = 5.0,
        vol_scale: float = 5.0,
    ):
        """
        Args:
            alpha: Temperature parameter for soft monotonicity (sigmoid sharpness)
            beta: Temperature parameter for soft curvature (sigmoid sharpness)
            periods: List of periods to analyze for periodicity scores.
                    Default: [2, 4, 8] (common cycles in time series)
            include_gbm: If True, append GBM drift and volatility concepts.
            drift_scale: Sensitivity scale for tanh normalization of GBM drift.
            vol_scale: Sensitivity scale for exponential normalization of GBM volatility.
        """
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.periods = periods if periods is not None else [2, 4, 8]
        self.include_gbm = include_gbm
        self.drift_scale = drift_scale
        self.vol_scale = vol_scale

        # K = 4 (mu_signed, mu_mag, kappa_signed, tau) + len(periods) [+ 2 if GBM]
        self.num_concepts = 4 + len(self.periods) + (2 if include_gbm else 0)

    @property
    def concept_names(self) -> List[str]:
        """Return labels for each concept dimension in output order."""
        names = [
            "mu_signed",    # index 0: signed monotonicity
            "mu_mag",       # index 1: magnitude of monotonicity
            "kappa_signed", # index 2: signed curvature
            "tau",          # index 3: observed tendency
        ]
        for p in self.periods:
            names.append(f"rho_p{p}")
        if self.include_gbm:
            names.append("gbm_drift")  # estimated GBM drift rate
            names.append("gbm_vol")    # estimated GBM volatility
        return names

    def forward(self, s: Tensor) -> Tensor:
        """
        Compute full temporal concept vector for batched segments.

        Args:
            s: Input sequences of shape (B, L)

        Returns:
            Concept vector of shape (B, K) where:
              K = 4 + len(periods)          when include_gbm=False
              K = 4 + len(periods) + 2      when include_gbm=True

            Column order:
              [mu_signed, mu_mag, kappa_signed, tau, rho_p*, gbm_drift*, gbm_vol*]
              (* gbm_drift and gbm_vol only present when include_gbm=True)

            Value ranges:
              mu_signed:   (-1, 1)
              mu_mag:      (0, 1)
              kappa_signed: (-1, 1)
              tau:         (-1, 1)
              rho_p*:      [0, 1]
              gbm_drift:   (-1, 1)
              gbm_vol:     [0, 1)
        """
        if s.dim() != 2:
            raise ValueError(f"Expected input of shape (B, L), got {tuple(s.shape)}")

        # T-01: Soft monotonicity
        mono_scores = soft_monotonicity(s, alpha=self.alpha)

        # T-02: Soft curvature
        curv_scores = soft_curvature(s, mu_signed=mono_scores.mu_signed, beta=self.beta)

        # T-03: Periodicity scores
        rho = periodicity_score(s, periods=self.periods)  # (B, len(periods))

        concept_vector = torch.stack([
            mono_scores.mu_signed,
            mono_scores.mu_mag,
            curv_scores.kappa_signed,
            curv_scores.tau,
        ], dim=1)  # (B, 4)

        concept_vector = torch.cat([concept_vector, rho], dim=1)  # (B, 4 + len(periods))

        # GBM stochastic concepts (optional)
        if self.include_gbm:
            stoch = gbm_scores(s, drift_scale=self.drift_scale, vol_scale=self.vol_scale)
            concept_vector = torch.cat([
                concept_vector,
                stoch.gbm_drift.unsqueeze(1),  # (B, 1)
                stoch.gbm_vol.unsqueeze(1),    # (B, 1)
            ], dim=1)  # (B, K+2)

        return concept_vector
