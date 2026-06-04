"""Standalone test-set evaluation from a saved TCRP checkpoint.

Run from the project root:
    python -m pipelines.evaluate \\
        --config pipelines/configs/etth1.yaml \\
        --checkpoint checkpoints/ETTh1_T336_H96_best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tcrp.dataset.preprocessing import inverse_transform
from tcrp.model.tcrp_forecaster.forecaster import TCRPConfig, TCRPForecaster

from .config import PipelineConfig, load_config
from .train import build_loaders


def evaluate(cfg: PipelineConfig, checkpoint: str) -> dict:
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = cfg.run_name or f"{cfg.dataset}_T{cfg.T}_H{cfg.H}"

    _, _, test_loader, mean, std = build_loaders(cfg)

    model_cfg = TCRPConfig(
        T=cfg.T, H=cfg.H, L=cfg.L, stride=cfg.stride,
        d=cfg.d, periods=cfg.periods,
        alpha=cfg.alpha, beta=cfg.beta,
        lambda1=cfg.lambda1, lambda2=cfg.lambda2,
        probabilistic=cfg.probabilistic,
    )
    model = TCRPForecaster(model_cfg).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x.to(device))
            preds.append(inverse_transform(out.y_hat.cpu(), mean, std))
            targets.append(inverse_transform(y, mean, std))

    p = torch.cat(preds)   # (N, H)
    t = torch.cat(targets) # (N, H)

    mse  = float(torch.mean((p - t) ** 2))
    mae  = float(torch.mean(torch.abs(p - t)))
    rmse = mse ** 0.5

    # Per-horizon step metrics
    per_h_mse = torch.mean((p - t) ** 2, dim=0).tolist()
    per_h_mae = torch.mean(torch.abs(p - t), dim=0).tolist()

    results = {
        "run_name":       run_name,
        "checkpoint":     checkpoint,
        "dataset":        cfg.dataset,
        "T":              cfg.T,
        "H":              cfg.H,
        "test_samples":   len(test_loader.dataset),
        "test_mse":       round(mse,  6),
        "test_mae":       round(mae,  6),
        "test_rmse":      round(rmse, 6),
        "per_horizon_mse": [round(v, 6) for v in per_h_mse],
        "per_horizon_mae": [round(v, 6) for v in per_h_mae],
    }

    print(f"\n{'=' * 55}")
    print(f"  Evaluation:   {run_name}")
    print(f"  Checkpoint:   {checkpoint}")
    print(f"  Test samples: {results['test_samples']:,}")
    print(f"  MSE:  {mse:.4f}   MAE: {mae:.4f}   RMSE: {rmse:.4f}")
    print(f"{'=' * 55}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved TCRP checkpoint on the test set."
    )
    parser.add_argument("--config",     required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt file")
    parser.add_argument("--out",        default=None,  help="Write JSON results here")
    args = parser.parse_args()

    cfg     = load_config(args.config)
    results = evaluate(cfg, args.checkpoint)

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()
