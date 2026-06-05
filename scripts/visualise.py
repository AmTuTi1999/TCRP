r"""TCRP analysis visualiser — CLI wrapper.

Four publication-style figures from a single look-back window:

    Plot 1 — R_x overlaid on raw input x          (shared x-axis)
    Plot 2 — Stacked area of R_x_cond per concept  (concept-conditional temporal maps)
    Plot 3 — R_h bar chart with sign               (concept relevance vector)
    Plot 4 — R_A heatmap (N × K)                  (segment × concept relevance)

Usage (run from project root):
    python scripts/visualise.py \
        --config  tcrp/pipelines/configs/etth1.yaml \
        [--checkpoint checkpoints/ETTh1_T336_H96_best.pt] \
        [--sample-idx 0] \
        [--h-star    0] \
        [--run-id    my_run]

Output is written to  figures/{run_id}/  as four PNG files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from tcrp.analysis.tcrp_analysis import TCRPAnalyser
from tcrp.analysis.visualise import plot_explanation
from tcrp.dataset.datasets import DATASET_META
from tcrp.model.tcrp_forecaster.forecaster import TCRPConfig, TCRPForecaster
from tcrp.pipelines.config import PipelineConfig, load_config
from tcrp.pipelines.train import build_loaders


def _find_checkpoint(cfg: PipelineConfig, explicit: str | None) -> str:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {explicit}")
        return str(p)
    run_name = cfg.run_name or f"{cfg.dataset}_T{cfg.T}_H{cfg.H}"
    candidate = Path(cfg.checkpoint_dir) / f"{run_name}_best.pt"
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(
        f"No checkpoint found at {candidate}.\n"
        f"Train first:  python -m tcrp.pipelines.train --config <yaml>\n"
        f"or pass      --checkpoint <path>"
    )


def _check_data(cfg: PipelineConfig) -> None:
    meta = DATASET_META.get(cfg.dataset)
    if meta is None:
        raise ValueError(f"Unknown dataset '{cfg.dataset}'")
    csv_path = Path(cfg.data_root) / meta["filename"]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found: {csv_path}\n"
            f"Download {cfg.dataset} and place it at that path."
        )


def visualise(
    cfg: PipelineConfig,
    checkpoint: str,
    sample_idx: int = 0,
    h_star: int = 0,
    run_id: str | None = None,
    highlight_seg: int | None = None,
) -> Path:
    """Load a checkpoint, run analysis on one test sample, and produce all five figures."""
    _check_data(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_id = run_id or cfg.run_name or f"{cfg.dataset}_T{cfg.T}_H{cfg.H}"
    out_dir = Path("figures") / run_id

    _, _, test_loader, _, _ = build_loaders(cfg)
    x_batch, _ = next(iter(test_loader))

    if sample_idx >= x_batch.shape[0]:
        raise IndexError(
            f"--sample-idx {sample_idx} >= batch size {x_batch.shape[0]}. "
            f"Use a value in [0, {x_batch.shape[0] - 1}]."
        )

    model_cfg = TCRPConfig(
        T=cfg.T,
        H=cfg.H,
        L=cfg.L,
        stride=cfg.stride,
        d=cfg.d,
        periods=cfg.periods,
        k_max=cfg.k_max,
        alpha=cfg.alpha,
        beta=cfg.beta,
        lambda1=cfg.lambda1,
        lambda2=cfg.lambda2,
        probabilistic=cfg.probabilistic,
    )
    model = TCRPForecaster(model_cfg).to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    model.eval()

    if h_star < 0 or h_star >= cfg.H:
        raise ValueError(f"--h-star {h_star} out of range [0, {cfg.H})")

    x_single = x_batch[sample_idx].unsqueeze(0).to(device)
    analyser = TCRPAnalyser(model, eps=1e-6)
    expl = analyser.analyse(x_single, h_star=h_star)

    return plot_explanation(
        expl,
        x_batch,
        concept_names=model.scorer.concept_names,
        run_id=run_id,
        h_star=h_star,
        out_dir=out_dir,
        sample_idx=sample_idx,
        highlight_seg=highlight_seg,
    )


def main() -> None:
    """Parse CLI arguments and produce all four TCRP analysis figures."""
    parser = argparse.ArgumentParser(
        description="TCRP analysis visualiser — four publication figures from a checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to YAML pipeline config")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to .pt checkpoint (auto-discovered if omitted)",
    )
    parser.add_argument(
        "--sample-idx", type=int, default=0, help="Index of test sample to visualise"
    )
    parser.add_argument(
        "--h-star",
        type=int,
        default=0,
        help="Forecast horizon step to explain (0-based)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Output folder name under figures/ (derived from config if omitted)",
    )
    parser.add_argument(
        "--highlight-seg",
        type=int,
        default=None,
        help="Segment index to drill down on in plot 5 (default: highest-relevance segment)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint = _find_checkpoint(cfg, args.checkpoint)

    out_dir = visualise(
        cfg,
        checkpoint,
        sample_idx=args.sample_idx,
        h_star=args.h_star,
        run_id=args.run_id,
        highlight_seg=args.highlight_seg,
    )
    print(f"Done. All figures in {out_dir}/")


if __name__ == "__main__":
    main()
