"""TCRP analysis utilities."""

from .lrp import lrp_gamma_conv, lrp_linear_eps, lrp_mean_pool, lrp_relu
from .narrator import TCRPNarrator
from .tcrp_analysis import TCRPAnalyser, TCRPExplanation, verify_conservation
from .visualise import (
    plot_concept_signal_vs_usage,
    plot_explanation,
    plot_segment_concept_map,
)

__all__ = [
    "lrp_linear_eps",
    "lrp_gamma_conv",
    "lrp_mean_pool",
    "lrp_relu",
    "TCRPAnalyser",
    "TCRPExplanation",
    "verify_conservation",
    "plot_explanation",
    "plot_segment_concept_map",
    "plot_concept_signal_vs_usage",
    "TCRPNarrator",
]
