"""Training pipeline entry point for TCRP.

Run from the project root:
    python -m pipelines.train --config pipelines/configs/etth1.yaml
    python -m pipelines.train --config pipelines/configs/etth1.yaml --H 192 --epochs 50
"""
from __future__ import annotations

import random, json, argparse
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from tcrp.dataset.datasets import TimeSeriesDataset, DATASET_META
from tcrp.dataset.preprocessing import inverse_transform
from tcrp.model.forecaster import TCRPConfig, TCRPForecaster
from tcrp.training.losses import LossBundle, TCRPLoss
from tcrp.training.trainer import Trainer

from .config import PipelineConfig, load_config


# ── Utilities ───────────────────────────────────────────────────────────────

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(cfg: PipelineConfig) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray]:
    """Return (train, val, test) DataLoaders plus train-split mean and std."""
    meta = DATASET_META[cfg.dataset]
    path = str(Path(cfg.data_root) / meta["filename"])

    # None → TimeSeriesDataset falls back to last numeric column
    target_col = cfg.target_col if cfg.univariate else None

    loader_kw = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    splits = {
        split: TimeSeriesDataset(
            path=path, split=split, T=cfg.T, H=cfg.H,
            normalise=True, target_col=target_col, univariate=cfg.univariate,
        )
        for split in ("train", "val", "test")
    }
    return (
        DataLoader(splits["train"], shuffle=True,  drop_last=False, **loader_kw),
        DataLoader(splits["val"],   shuffle=False, **loader_kw),
        DataLoader(splits["test"],  shuffle=False, **loader_kw),
        splits["train"].mean,
        splits["train"].std,
    )


def _eval_denorm(
    model: nn.Module,
    loader: DataLoader,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> dict:
    """Evaluate model, returning denormalised MSE / MAE / RMSE."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            preds.append(inverse_transform(out.y_hat.cpu(), mean, std))
            targets.append(inverse_transform(y, mean, std))
    p = torch.cat(preds)
    t = torch.cat(targets)
    mse  = float(torch.mean((p - t) ** 2))
    mae  = float(torch.mean(torch.abs(p - t)))
    return {"mse": round(mse, 6), "mae": round(mae, 6), "rmse": round(mse ** 0.5, 6)}


# ── PipelineTrainer ─────────────────────────────────────────────────────────

class PipelineTrainer(Trainer):
    """Extends Trainer with configurable LR, grad clipping, and ES patience."""

    def __init__(
        self,
        model: nn.Module,
        tcrp_cfg: TCRPConfig,
        pipeline_cfg: PipelineConfig,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(model, tcrp_cfg, device=device)
        # Override hardcoded defaults from parent
        for pg in self.optimizer.param_groups:
            pg["lr"] = pipeline_cfg.lr
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min",
            patience=pipeline_cfg.lr_patience,
            factor=pipeline_cfg.lr_factor,
        )
        self._grad_clip = pipeline_cfg.grad_clip
        self._es_patience = pipeline_cfg.early_stopping_patience

    # Override to add gradient clipping
    def train_epoch(self, loader: DataLoader) -> LossBundle:
        self.model.train()
        totals = {"forecast": 0.0, "align": 0.0, "reg": 0.0, "total": 0.0}
        count = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(x)
            lb = self.criterion(out.y_hat, y, out.A, out.C, self.model.projection)
            lb.total_loss.backward()
            if self._grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._grad_clip)
            self.optimizer.step()
            bs = x.shape[0]
            totals["forecast"] += lb.forecast_loss.item() * bs
            totals["align"]    += lb.align_loss.item()    * bs
            totals["reg"]      += lb.reg_loss.item()      * bs
            totals["total"]    += lb.total_loss.item()    * bs
            count += bs
        return LossBundle(
            forecast_loss=torch.tensor(totals["forecast"] / count, device=self.device),
            align_loss=   torch.tensor(totals["align"]    / count, device=self.device),
            reg_loss=     torch.tensor(totals["reg"]      / count, device=self.device),
            total_loss=   torch.tensor(totals["total"]    / count, device=self.device),
        )

    # Override to use configurable ES patience
    def fit(self, train_loader: DataLoader, val_loader: DataLoader, max_epochs: int = 100) -> None:
        for epoch in range(1, max_epochs + 1):
            train_lb = self.train_epoch(train_loader)
            val_m    = self.validate(val_loader)
            self.scheduler.step(val_m["mse"])
            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"{epoch:6d} | {train_lb.forecast_loss.item():10.6f} | "
                f"{val_m['mse']:10.6f} | {val_m['align_loss']:10.6f} | {lr:10.2e}"
            )
            if val_m["mse"] < self.best_val_mse:
                self.best_val_mse = val_m["mse"]
                self.epochs_no_improve = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                self.epochs_no_improve += 1
            if self.epochs_no_improve >= self._es_patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"({self.epochs_no_improve} epochs without improvement)"
                )
                break


# ── Main pipeline function ───────────────────────────────────────────────────

def run(cfg: PipelineConfig) -> dict:
    """Build data, model, and trainer; run training; evaluate on test set."""
    _seed_everything(cfg.seed)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = cfg.run_name or f"{cfg.dataset}_T{cfg.T}_H{cfg.H}"

    print(f"{'=' * 62}")
    print(f"  TCRP Training — {run_name}")
    print(f"  device={device}  seed={cfg.seed}  lr={cfg.lr}  clip={cfg.grad_clip}")
    print(f"{'=' * 62}")

    # ── Data ────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, mean, std = build_loaders(cfg)
    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset)
    n_test  = len(test_loader.dataset)
    print(
        f"Splits  train:{n_train:,}  val:{n_val:,}  test:{n_test:,}  "
        f"batches/epoch:{len(train_loader)}"
    )

    # ── Model ───────────────────────────────────────────────────────────
    model_cfg = TCRPConfig(
        T=cfg.T, H=cfg.H, L=cfg.L, stride=cfg.stride,
        d=cfg.d, periods=cfg.periods,
        alpha=cfg.alpha, beta=cfg.beta,
        lambda1=cfg.lambda1, lambda2=cfg.lambda2,
        probabilistic=cfg.probabilistic,
        include_gbm=cfg.include_gbm,
    )
    model = TCRPForecaster(model_cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_segs   = (cfg.T - cfg.L) // cfg.stride + 1
    print(f"Params  {n_params:,}  K={model_cfg.K}  N~{n_segs} segments/window")

    # ── Trainer ─────────────────────────────────────────────────────────
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    trainer = PipelineTrainer(model, model_cfg, cfg, device=device)
    trainer.checkpoint_path = str(ckpt_dir / f"{run_name}_best.pt")

    print(f"\n{'Epoch':>6} | {'train_MSE':>10} | {'val_MSE':>10} | {'align':>10} | {'lr':>10}")
    print("-" * 62)

    t0 = time.time()
    trainer.fit(train_loader, val_loader, max_epochs=cfg.max_epochs)
    elapsed = time.time() - t0

    # ── Test evaluation (denormalised) ───────────────────────────────────
    model.load_state_dict(
        torch.load(trainer.checkpoint_path, map_location=device, weights_only=True)
    )
    val_m  = _eval_denorm(model, val_loader,  mean, std, device)
    test_m = _eval_denorm(model, test_loader, mean, std, device)

    # ── Persist results ──────────────────────────────────────────────────
    results = {
        "run_name":          run_name,
        "dataset":           cfg.dataset,
        "T":                 cfg.T,
        "H":                 cfg.H,
        "n_params":          n_params,
        "train_time_s":      round(elapsed, 1),
        "best_val_mse_norm": round(trainer.best_val_mse, 6),
        "val_mse":           val_m["mse"],
        "val_mae":           val_m["mae"],
        "val_rmse":          val_m["rmse"],
        "test_mse":          test_m["mse"],
        "test_mae":          test_m["mae"],
        "test_rmse":         test_m["rmse"],
    }
    results_path = ckpt_dir / f"{run_name}_results.json"
    results_path.write_text(json.dumps(results, indent=2))

    print(f"\n{'─' * 40}")
    print(f"  val   MSE={val_m['mse']:.4f}  MAE={val_m['mae']:.4f}  RMSE={val_m['rmse']:.4f}")
    print(f"  test  MSE={test_m['mse']:.4f}  MAE={test_m['mae']:.4f}  RMSE={test_m['rmse']:.4f}")
    print(f"  time  {elapsed:.1f}s")
    print(f"  ckpt  {trainer.checkpoint_path}")
    print(f"  json  {results_path}")
    return results


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a TCRP model. Must be run from the project root."
    )
    parser.add_argument("--config",   required=True,  help="Path to YAML config")
    parser.add_argument("--dataset",  default=None,   help="Override dataset name")
    parser.add_argument("--T",        type=int,       help="Look-back window")
    parser.add_argument("--H",        type=int,       help="Forecast horizon")
    parser.add_argument("--epochs",   type=int,       help="Max training epochs")
    parser.add_argument("--lr",       type=float,     help="Learning rate")
    parser.add_argument("--batch",    type=int,       help="Batch size")
    parser.add_argument("--seed",     type=int,       help="Random seed")
    parser.add_argument("--run-name", default=None,   help="Checkpoint/results filename prefix")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset:  cfg.dataset    = args.dataset
    if args.T:        cfg.T          = args.T
    if args.H:        cfg.H          = args.H
    if args.epochs:   cfg.max_epochs = args.epochs
    if args.lr:       cfg.lr         = args.lr
    if args.batch:    cfg.batch_size = args.batch
    if args.seed:     cfg.seed       = args.seed
    if args.run_name: cfg.run_name   = args.run_name

    run(cfg)


if __name__ == "__main__":
    main()
