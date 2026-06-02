"""
T-04 · Full Temporal Concept Vector
Paper: Eq. 26, Table 2

Combines all Phase-1 analytic concept scores into a single differentiable vector.
No learned parameters — pure analytic computation.

Canonical column order (K = 16 + k_max + len(periods)):
  [0:2]              mu_signed, mu_mag            — T-01 trend
  [2:4]              kappa_signed, tau             — T-02 curvature
  [4]                xi                            — stochasticity (GBM vol)
  [5:8]              sigma_tilde, mu_v, psi        — T-03c volatility
  [8:8+k_max]        rho_1..rho_k_max              — T-03d increment ACF
  [8+k:10+k]         theta_hat, z                  — T-03d mean reversion / z-score
  [10+k:13+k]        b_mu, b_sigma, b_mu_tilde     — T-03e structural breaks
  [13+k:16+k]        varsigma, kappa4, j           — T-03f distributional shape
  [16+k:]            rho_p1..rho_pM                — T-03 periodicity

Default K = 16 + 2 + 4 = 22  (k_max=2, periods=[2,4,8,16]).
"""

from typing import List
import torch
import torch.nn as nn
from torch import Tensor

from .monotonicity import soft_monotonicity
from .curvature import soft_curvature
from .periodicity import periodicity_score
from .stochastic import gbm_scores
from .volatility import volatility_scores
from .autocorrelation import autocorrelation_scores
from .structural_breaks import break_scores
from .distribution_shape import shape_scores


class ConceptScorer(nn.Module):
    """
    Computes the full temporal concept vector for time series segments.

    No learned parameters — all computations are fully differentiable
    w.r.t. the input.

    Args:
        alpha:           Sigmoid sharpness for soft monotonicity / volatility trend.
        beta:            Sigmoid sharpness for soft curvature.
        periods:         Periods (samples) for spectral periodicity scores.
                         Default [2, 4, 8, 16] → len(periods)=4 → K=22.
        gamma:           Vol-scale sensitivity for stochasticity concept xi.
        gamma_j:         Sigmoid sharpness for soft jump indicator j.
        k_max:           Number of ACF lags to compute.
        jump_threshold:  Max-increment threshold (multiples of std) for j.
        train_std:       Training-set std for normalising realised volatility.
    """

    def __init__(
        self,
        alpha: float = 5.0,
        beta: float = 5.0,
        periods: List[int] = None,
        gamma: float = 0.5,
        gamma_j: float = 2.0,
        k_max: int = 2,
        jump_threshold: float = 3.0,
        train_std: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.periods = periods if periods is not None else [2, 4, 8, 16]
        self.gamma = gamma
        self.gamma_j = gamma_j
        self.k_max = k_max
        self.jump_threshold = jump_threshold
        self.train_std = train_std
        self.num_concepts = 16 + k_max + len(self.periods)

    @property
    def concept_names(self) -> List[str]:
        """Return label for each column of the concept vector, in output order."""
        k = self.k_max
        names: List[str] = [
            "mu_signed", "mu_mag",           # T-01 trend
            "kappa_signed", "tau",            # T-02 curvature
            "xi",                             # stochasticity
            "sigma_tilde", "mu_v", "psi",    # T-03c volatility
        ]
        for lag in range(1, k + 1):
            names.append(f"rho_{lag}")        # T-03d increment ACF
        names += ["theta_hat", "z"]           # T-03d mean reversion / z-score
        names += ["b_mu", "b_sigma", "b_mu_tilde"]   # T-03e breaks
        names += ["varsigma", "kappa4", "j"] # T-03f shape
        for p in self.periods:
            names.append(f"rho_p{p}")         # T-03 periodicity
        assert len(names) == self.num_concepts, (
            f"concept_names length {len(names)} != num_concepts {self.num_concepts}"
        )
        return names

    def forward(self, s: Tensor) -> Tensor:
        """
        Compute full temporal concept vector for a batch of segments.

        Args:
            s: Input of shape (B, L).

        Returns:
            Concept vector of shape (B, K).
        """
        if s.dim() != 2:
            raise ValueError(f"Expected (B, L), got shape {tuple(s.shape)}")

        # T-01: Soft monotonicity
        mono = soft_monotonicity(s, alpha=self.alpha)

        # T-02: Soft curvature
        curv = soft_curvature(s, mu_signed=mono.mu_signed, beta=self.beta)

        # xi: stochasticity — GBM volatility estimate with gamma as sensitivity scale
        stoch = gbm_scores(s, vol_scale=self.gamma)
        xi = stoch.gbm_vol                                   # (B,)

        # T-03c: Volatility
        vol = volatility_scores(s, train_std=self.train_std, alpha=self.alpha)

        # T-03d: Autocorrelation (rho, theta_hat, z — all enabled)
        acf = autocorrelation_scores(s, k_max=self.k_max)

        # T-03e: Structural breaks
        brk = break_scores(s, alpha=self.alpha)

        # T-03f: Distributional shape
        shp = shape_scores(s, gamma_j=self.gamma_j, jump_threshold=self.jump_threshold)

        # T-03: Periodicity
        rho_p = periodicity_score(s, periods=self.periods)  # (B, M)

        # Assemble in canonical order
        parts = [
            mono.mu_signed.unsqueeze(1),        # (B,1)
            mono.mu_mag.unsqueeze(1),
            curv.kappa_signed.unsqueeze(1),
            curv.tau.unsqueeze(1),
            xi.unsqueeze(1),
            vol["sigma_tilde"].unsqueeze(1),
            vol["mu_v"].unsqueeze(1),
            vol["psi"].unsqueeze(1),
            acf["rho"],                          # (B, k_max)
            acf["theta_hat"].unsqueeze(1),
            acf["z"].unsqueeze(1),
            brk["b_mu"].unsqueeze(1),
            brk["b_sigma"].unsqueeze(1),
            brk["b_mu_tilde"].unsqueeze(1),
            shp["varsigma"].unsqueeze(1),
            shp["kappa4"].unsqueeze(1),
            shp["j"].unsqueeze(1),
            rho_p,                               # (B, M)
        ]
        return torch.cat(parts, dim=1)           # (B, K)
