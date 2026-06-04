"""Phase 2: Model Architecture.

Segmentation and encoder modules for temporal feature extraction.

Exports:
  - Segmenter: Sliding window segmentation (T-05)
  - CausalDilatedBlock: Dilated causal TCN block (T-06)
"""

from .baselines import (
    BaselineOutput,
    LSTMForecaster,
    NBeats,
    TCNForecaster,
    build_baseline,
)
from .configs import (
    ConceptAttentionPoolConfig,
    ConceptScorerConfig,
    GaussianDecoderConfig,
    HorizonDecoderConfig,
    LSTMEncoderConfig,
    SegmenterConfig,
    TCNEncoderConfig,
)
from .tcrp_forecaster.components.adversarial import (
    AdversarialTCRPForecaster,
    GRLLayer,
    grl_alpha_schedule,
)
from .tcrp_forecaster.components.aggregation import ConceptAttentionPool
from .tcrp_forecaster.components.bottleneck import (
    ConceptProjection,
    alignment_loss,
    stability_loss,
)
from .tcrp_forecaster.components.decoder import (
    BaseDecoder,
    GaussianDecoder,
    HorizonDecoder,
)
from .tcrp_forecaster.components.encoder import (
    BaseEncoder,
    CausalDilatedBlock,
    LSTMEncoder,
    TCNEncoder,
)
from .tcrp_forecaster.components.segmentation import Segmenter
from .tcrp_forecaster.forecaster import TCRPConfig, TCRPForecaster, TCRPOutput

__all__ = [
    # Components
    "Segmenter",
    "BaseEncoder",
    "CausalDilatedBlock",
    "TCNEncoder",
    "LSTMEncoder",
    "ConceptProjection",
    "alignment_loss",
    "stability_loss",
    "ConceptAttentionPool",
    "BaseDecoder",
    "HorizonDecoder",
    "GaussianDecoder",
    # Core model
    "TCRPConfig",
    "TCRPForecaster",
    "TCRPOutput",
    # Adversarial
    "GRLLayer",
    "grl_alpha_schedule",
    "AdversarialTCRPForecaster",
    # Baselines
    "NBeats",
    "LSTMForecaster",
    "TCNForecaster",
    "BaselineOutput",
    "build_baseline",
    # Component configs
    "TCNEncoderConfig",
    "LSTMEncoderConfig",
    "HorizonDecoderConfig",
    "GaussianDecoderConfig",
    "SegmenterConfig",
    "ConceptScorerConfig",
    "ConceptAttentionPoolConfig",
]
