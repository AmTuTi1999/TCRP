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
  - autocorrelation_scores: Lag-k ACF, mean-reversion speed, z-score (T-03d)
  - AutocorrelationScores: TypedDict for autocorrelation outputs
"""

from .monotonicity import soft_monotonicity, MonotonicityScores
from .curvature import soft_curvature, CurvatureScores
from .periodicity import periodicity_score
from .concept_vector import ConceptScorer
from .stochastic import gbm_scores, GBMScores
from .autocorrelation import autocorrelation_scores, AutocorrelationScores
from .volatility import volatility_scores, VolatilityScores
from .distribution_shape import shape_scores, ShapeScores
from .structural_breaks import break_scores, BreakScores

__all__ = [
    "soft_monotonicity",
    "MonotonicityScores",
    "soft_curvature",
    "CurvatureScores",
    "periodicity_score",
    "ConceptScorer",
    "gbm_scores",
    "GBMScores",
    "autocorrelation_scores",
    "AutocorrelationScores",
    "volatility_scores",
    "VolatilityScores",
    "shape_scores",
    "ShapeScores",
    "break_scores",
    "BreakScores",
]
