"""
Phase 1: Analytic Concept Scores
Pure functions and modules for computing differentiable temporal concepts.

Exports:
  - soft_monotonicity: Soft monotonicity score (T-01)
  - soft_curvature: Soft curvature and observed tendency (T-02)
  - periodicity_score: Periodicity scores at multiple periods (T-03)
  - ConceptScorer: Full concept vector computation (T-04)
  - gbm_scores: GBM drift and volatility estimation (stochastic)
  - GBMScores: NamedTuple for GBM parameter estimates
"""

from .monotonicity import soft_monotonicity, MonotonicityScores
from .curvature import soft_curvature, CurvatureScores
from .periodicity import periodicity_score
from .concept_vector import ConceptScorer
from .stochastic import gbm_scores, GBMScores

__all__ = [
    "soft_monotonicity",
    "MonotonicityScores",
    "soft_curvature",
    "CurvatureScores",
    "periodicity_score",
    "ConceptScorer",
    "gbm_scores",
    "GBMScores",
]
