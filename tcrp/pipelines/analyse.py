"""TCRP analysis pipeline — relevance propagation pass on a saved checkpoint.

Run from the project root:
    python -m tcrp.pipelines.analyse \\
        --config tcrp/pipelines/configs/etth1.yaml \\
        --checkpoint checkpoints/ETTh1_T336_H96_best.pt

Optional flags:
    --h-star 0          Forecast horizon step to explain (default: 0 = first)
    --n-samples 32      Number of test samples to run (default: 32)
    --out path/to.pt    Save explanation tensors here (default: print only)
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch

from tcrp.analysis.tcrp_analysis import TCRPAnalyser, TCRPExplanation, verify_conservation
from tcrp.model.tcrp_forecaster.forecaster import TCRPConfig, TCRPForecaster

from tcrp.dataset.datasets import DATASET_META

from .config import PipelineConfig, load_config
from .train import build_loaders


def _check_data(cfg: PipelineConfig) -> None:
    """Raise FileNotFoundError with a clear message if the dataset CSV is missing."""
    meta = DATASET_META.get(cfg.dataset)
    if meta is None:
        raise ValueError(f"Unknown dataset '{cfg.dataset}'")
    csv_path = Path(cfg.data_root) / meta["filename"]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {csv_path}\n"
            f"Download {cfg.dataset} and place it at {csv_path}"
        )


def analyse(
    cfg: PipelineConfig,
    checkpoint: str,
    h_star: int = 0,
    n_samples: int = 32,
) -> dict:
    """Run the TCRP analysis pass and return a results dict.

    Args:
        cfg: Pipeline configuration.
        checkpoint: Path to a saved model state-dict (.pt file).
        h_star: Forecast horizon step index to explain (0-based).
        n_samples: Number of test-set samples to analyse.

    Returns:
        Dict with keys:
            run_name, h_star, n_samples, conserved, concept_names,
            explanation (dict of CPU tensors matching TCRPExplanation fields).
    """
    _check_data(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = cfg.run_name or f"{cfg.dataset}_T{cfg.T}_H{cfg.H}"

    _, _, test_loader, _, _ = build_loaders(cfg)

    model_cfg = TCRPConfig(
        T=cfg.T, H=cfg.H, L=cfg.L, stride=cfg.stride,
        d=cfg.d, periods=cfg.periods,
        k_max=cfg.k_max,
        alpha=cfg.alpha, beta=cfg.beta,
        lambda1=cfg.lambda1, lambda2=cfg.lambda2,
        probabilistic=cfg.probabilistic,
    )
    model = TCRPForecaster(model_cfg).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    if h_star < 0 or h_star >= cfg.H:
        raise ValueError(f"h_star={h_star} out of range [0, {cfg.H})")

    # Grab the first batch from the test loader (up to n_samples)
    x_batch, _ = next(iter(test_loader))
    x_batch = x_batch[:n_samples].to(device)

    analyser = TCRPAnalyser(model, eps=1e-6)
    explanation = analyser.analyse(x_batch, h_star=h_star)

    with torch.no_grad():
        out = model(x_batch)

    conserved = verify_conservation(explanation, out.y_hat, h_star=h_star, tol=1e-3)

    # Summarise concept relevances
    concept_names = model.scorer.concept_names
    R_h_mean = explanation.R_h.abs().mean(0)          # (K,)
    top_n = min(5, len(concept_names))
    top_k = R_h_mean.topk(top_n)

    print(f"\n{'=' * 60}")
    print(f"  TCRP Analysis — {run_name}")
    print(f"  checkpoint : {checkpoint}")
    print(f"  h_star     : {h_star}  (horizon step {h_star + 1}/{cfg.H})")
    print(f"  samples    : {x_batch.shape[0]}")
    print(f"  conservation: {'PASS' if conserved else 'FAIL (check tol)'}")
    print(f"\n  Top-{top_n} concept relevances (mean |R_h| over batch):")
    for rank, (idx, val) in enumerate(
        zip(top_k.indices.tolist(), top_k.values.tolist()), start=1
    ):
        name = concept_names[idx] if idx < len(concept_names) else f"concept_{idx}"
        print(f"    {rank}. [{idx:2d}] {name:<32s}  {val:.6f}")
    print(f"{'=' * 60}\n")

    return {
        "run_name":      run_name,
        "h_star":        h_star,
        "n_samples":     x_batch.shape[0],
        "conserved":     conserved,
        "concept_names": concept_names,
        "explanation": {
            "R_h":      explanation.R_h.cpu(),
            "R_A":      explanation.R_A.cpu(),
            "R_x":      explanation.R_x.cpu(),
            "R_x_cond": explanation.R_x_cond.cpu(),
            "eta":      explanation.eta.cpu(),
            "A":        explanation.A.cpu(),
            "C":        explanation.C.cpu(),
        },
    }


def _find_checkpoint(cfg: PipelineConfig, explicit: Optional[str]) -> str:
    """Resolve a checkpoint path.

    Priority:
        1. Explicit --checkpoint argument.
        2. Auto-discover: {checkpoint_dir}/{run_name}_best.pt using the same
           naming convention as the training pipeline.

    Raises FileNotFoundError with a helpful message if nothing is found.
    """
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
        f"No checkpoint found. Tried: {candidate}\n"
        f"Either train first with:\n"
        f"  python -m tcrp.pipelines.train --config {cfg.dataset.lower()}.yaml\n"
        f"or pass --checkpoint <path> explicitly."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TCRP relevance propagation on a saved checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",     required=True, help="Path to YAML pipeline config")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to saved .pt state-dict (auto-discovered from config if omitted)")
    parser.add_argument("--h-star",     type=int, default=0,  help="Horizon step to explain (0-based)")
    parser.add_argument("--n-samples",  type=int, default=32, help="Number of test samples to analyse")
    parser.add_argument("--out",        default=None,
                        help="Save explanation tensors to this .pt file (optional)")
    args = parser.parse_args()

    cfg        = load_config(args.config)
    checkpoint = _find_checkpoint(cfg, args.checkpoint)
    result     = analyse(cfg, checkpoint, h_star=args.h_star, n_samples=args.n_samples)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result["explanation"], str(out_path))
        print(f"Explanation saved → {out_path}")


if __name__ == "__main__":
    main()
