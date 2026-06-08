"""Training script entry point for running the TCRP training pipeline via Hydra."""

import os
import sys

import hydra
from omegaconf import DictConfig

from tcrp.pipelines.train import run


# ── CLI ─────────────────────────────────────────────────────────────────────
@hydra.main(version_base=None, config_path="configs", config_name="train_baselines")
def main(cfg: DictConfig):
    """Run the heart disease training pipeline.

    Args:
        cfg (DictConfig): Configuration for the experiment including model, data, and cross-validation parameters.
    """
    # Initialize and run the training pipeline
    run(cfg)
    # Run the main function


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"  # Set this to get full error messages
    print(sys.path)
    main()
