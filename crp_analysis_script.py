"""CRP analysis script entry point for running TCRP relevance propagation via Hydra."""

import os
import sys

import hydra
from omegaconf import DictConfig

from tcrp.pipelines.analyse import analyse


# ── CLI ─────────────────────────────────────────────────────────────────────
@hydra.main(version_base=None, config_path="configs", config_name="crp_analysis")
def main(cfg: DictConfig):
    """Run the heart disease training pipeline.

    Args:
        cfg (DictConfig): Configuration for the experiment including model, data, and cross-validation parameters.
    """
    # Initialize and run the training pipeline
    analyse(cfg)
    # Run the main function


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"  # Set this to get full error messages
    print(sys.path)
    main()
