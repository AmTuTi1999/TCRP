"""
Phase 2: Model Architecture

Segmentation and encoder modules for temporal feature extraction.

Exports:
  - Segmenter: Sliding window segmentation (T-05)
  - CausalDilatedBlock: Dilated causal TCN block (T-06)
"""

from .segmentation import Segmenter
from .encoder import CausalDilatedBlock, TCNEncoder, LSTMEncoder
from .bottleneck import ConceptProjection, alignment_loss
from .aggregation import ConceptAttentionPool
from .decoder import HorizonDecoder, GaussianDecoder
from .forecaster import TCRPConfig, TCRPForecaster, TCRPOutput

__all__ = [
    "Segmenter",
    "CausalDilatedBlock",
    "TCNEncoder",
    "LSTMEncoder",
    "ConceptProjection",
    "alignment_loss",
    "ConceptAttentionPool",
    "HorizonDecoder",
    "GaussianDecoder",
    "TCRPConfig",
    "TCRPForecaster",
    "TCRPOutput",
]
