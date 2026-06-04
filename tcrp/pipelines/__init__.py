"""Training and analysis pipeline entry points for TCRP."""

from .config import PipelineConfig, load_config
from .evaluate import evaluate
from .train import run

__all__ = ["PipelineConfig", "load_config", "evaluate", "run"]
