"""Training pipeline entry point for TCRP.

Run from the project root:
    python -m pipelines.train --config pipelines/configs/etth1.yaml
    python -m pipelines.train --config pipelines/configs/etth1.yaml --H 192 --epochs 50
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from tcrp.dataset.datasets import DATASET_META, TimeSeriesDataset
from tcrp.model.baselines import build_baseline
from tcrp.model.tcrp_forecaster.components.adversarial import AdversarialTCRPForecaster
from tcrp.model.tcrp_forecaster.forecaster import TCRPConfig, TCRPForecaster
from tcrp.pipelines.config import tcrp_config_from_hydra
from tcrp.training.adversarial_trainer import AdversarialTrainer
from tcrp.training.baseline_trainer import BaselineTrainer
from tcrp.training.losses import LossBundle
from tcrp.training.trainer import Trainer
from tcrp.utils import elapsed_str, eval_denorm, now_iso, save_results, seed_everything


def build_loaders(
    cfg: DictConfig,
) -> tuple[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray]:
    """Return (train, val, test) DataLoaders plus train-split mean and std."""
    ds = cfg.datasets
    tr = cfg.trainers
    meta = DATASET_META[ds.dataset]
    path = str(Path(ds.data_root) / meta["filename"])
    target_col = ds.target_col if ds.univariate else None
    loader_kw = {
        "batch_size": tr.batch_size,
        "num_workers": tr.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    splits = {
        split: TimeSeriesDataset(
            path=path,
            split=split,
            T=cfg.T,
            H=cfg.H,
            normalise=True,
            target_col=target_col,
            univariate=ds.univariate,
        )
        for split in ("train", "val", "test")
    }
    return (
        DataLoader(splits["train"], shuffle=True, drop_last=False, **loader_kw),
        DataLoader(splits["val"], shuffle=False, **loader_kw),
        DataLoader(splits["test"], shuffle=False, **loader_kw),
        splits["train"].mean,
        splits["train"].std,
    )


class PipelineTrainer(Trainer):
    """Extends Trainer with configurable LR, grad clipping, and ES patience."""

    def __init__(
        self,
        model: nn.Module,
        tcrp_cfg: TCRPConfig,
        pipeline_cfg: DictConfig,
        device: torch.device | None = None,
    ) -> None:
        """Initialize PipelineTrainer with configurable LR, grad clip, and ES patience."""
        super().__init__(model, tcrp_cfg, device=device)
        # Override hardcoded defaults from parent
        for pg in self.optimizer.param_groups:
            pg["lr"] = pipeline_cfg.lr
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=pipeline_cfg.lr_patience,
            factor=pipeline_cfg.lr_factor,
        )
        self._grad_clip = pipeline_cfg.grad_clip
        self._es_patience = pipeline_cfg.early_stopping_patience

    # Override to add gradient clipping
    def train_epoch(self, loader: DataLoader) -> LossBundle:
        """Run one training epoch with gradient clipping."""
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
            totals["align"] += lb.align_loss.item() * bs
            totals["reg"] += lb.reg_loss.item() * bs
            totals["total"] += lb.total_loss.item() * bs
            count += bs
        return LossBundle(
            forecast_loss=torch.tensor(totals["forecast"] / count, device=self.device),
            align_loss=torch.tensor(totals["align"] / count, device=self.device),
            reg_loss=torch.tensor(totals["reg"] / count, device=self.device),
            total_loss=torch.tensor(totals["total"] / count, device=self.device),
            mag_loss=None,
            stab_loss=None,
        )

    # Override to use configurable ES patience
    def fit(
        self, train_loader: DataLoader, val_loader: DataLoader, max_epochs: int = 100
    ) -> None:
        """Fit the model with configurable early-stopping patience."""
        for epoch in range(1, max_epochs + 1):
            train_lb = self.train_epoch(train_loader)
            val_m = self.validate(val_loader)
            self.scheduler.step(val_m["mse"])
            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"{epoch:6d} | {train_lb.forecast_loss.item():10.6f} | "
                f"{val_m['mse']:10.6f} | {val_m['align']:10.6f} | {lr:10.2e}"
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


class AdversarialPipelineTrainer(AdversarialTrainer):
    """Extends AdversarialTrainer with configurable LR, grad clipping, ES patience."""

    def __init__(
        self,
        model: AdversarialTCRPForecaster,
        tcrp_cfg: TCRPConfig,
        pipeline_cfg: DictConfig,
        device: torch.device | None = None,
    ) -> None:
        """Initialize AdversarialPipelineTrainer with configurable LR, grad clip, and ES patience."""
        super().__init__(model, tcrp_cfg, device=device)
        for pg in self.optimizer.param_groups:
            pg["lr"] = pipeline_cfg.lr
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=pipeline_cfg.lr_patience,
            factor=pipeline_cfg.lr_factor,
        )
        self._es_patience = pipeline_cfg.early_stopping_patience

    def fit(
        self, train_loader: DataLoader, val_loader: DataLoader, max_epochs: int = 100
    ) -> None:
        """Fit the adversarial model with GRL alpha scheduling and early stopping."""
        from tcrp.model.tcrp_forecaster.components.adversarial import grl_alpha_schedule

        for epoch in range(1, max_epochs + 1):
            alpha = grl_alpha_schedule(
                epoch - 1,
                max_epochs,
                warmup_epochs=self.config.warmup_epochs,
                alpha_max=self.config.alpha_max,
            )
            self.model.set_alpha(alpha)

            train_lb = self.train_epoch(train_loader)
            val_m = self.validate(val_loader)
            self.scheduler.step(val_m["mse"])
            lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"{epoch:6d} | α={alpha:.3f} | "
                f"fc {train_lb.forecast_loss.item():.5f} | "
                f"aln {train_lb.align_loss.item():.5f} | "
                f"mag {train_lb.mag_loss.item():.5f} | "
                f"stb {train_lb.stab_loss.item():.5f} | "
                f"val_mse {val_m['mse']:.5f} | "
                f"val_mag {val_m['mag']:.5f} | "
                f"lr {lr:.2e}"
            )

            if val_m["mse"] < self.best_val_mse:
                self.best_val_mse = val_m["mse"]
                self.epochs_no_improve = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                self.epochs_no_improve += 1
            if self.epochs_no_improve >= self._es_patience:
                print(
                    f"Early stopping at epoch {epoch} ({self.epochs_no_improve} epochs without improvement)"
                )
                break


_BASELINE_TYPES = {"nbeats", "lstm", "tcn"}


def run(cfg: DictConfig) -> dict:
    """Build data, model, and trainer; run training; evaluate on test set."""
    seed_everything(cfg.seed)
    dataset_cfg = cfg.datasets
    trainer_cfg = cfg.trainers
    model_cfg = cfg.models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = (
        cfg.run_name or f"{dataset_cfg.dataset}_{cfg.model_type}_T{cfg.T}_H{cfg.H}"
    )

    is_baseline = cfg.model_type.lower() in _BASELINE_TYPES
    mode_str = cfg.model_type + (
        "-adv" if model_cfg.adversarial and not is_baseline else ""
    )
    print(f"{'=' * 66}")
    print(f"  Training [{mode_str}] — {run_name}")
    print(
        f"  device={device}  seed={trainer_cfg.seed}  lr={trainer_cfg.lr}  clip={trainer_cfg.grad_clip}"
    )
    if model_cfg.adversarial and not is_baseline:
        print(
            f"  alpha_max={model_cfg.alpha_max}  warmup={model_cfg.warmup_epochs}  lambda3={model_cfg.lambda3}"
        )
    if is_baseline:
        print(
            f"  hidden={model_cfg.baseline_hidden}  layers={model_cfg.baseline_layers}"
        )
    print(f"{'=' * 66}")

    # ── Data ────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, mean, std = build_loaders(cfg)
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    n_test = len(test_loader.dataset)
    print(
        f"Splits  train:{n_train:,}  val:{n_val:,}  test:{n_test:,}  "
        f"batches/epoch:{len(train_loader)}"
    )

    # ── Model ───────────────────────────────────────────────────────────
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    mode_tag = "_adv" if (not is_baseline and model_cfg.adversarial) else "_std"
    ckpt_path = str(ckpt_dir / f"{run_name}{mode_tag}_best.pt")

    if is_baseline:
        model = build_baseline(
            cfg.model_type,
            cfg.T,
            cfg.H,
            hidden=model_cfg.baseline_hidden,
            n_layers=model_cfg.baseline_layers,
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Params  {n_params:,}")

        trainer = BaselineTrainer(
            model,
            lr=trainer_cfg.lr,
            lr_patience=trainer_cfg.lr_patience,
            lr_factor=trainer_cfg.lr_factor,
            grad_clip=trainer_cfg.grad_clip,
            es_patience=trainer_cfg.early_stopping_patience,
            checkpoint_path=ckpt_path,
            device=device,
        )
        header = f"\n{'Epoch':>6} | {'train_MSE':>10} | {'val_MSE':>10} | {'val_MAE':>10} | {'lr':>10}"
        sep = "-" * 58

    else:
        model_cfg: TCRPConfig = tcrp_config_from_hydra(cfg)
        base_model = TCRPForecaster(model_cfg)
        if model_cfg.adversarial:
            model = AdversarialTCRPForecaster(base_model, alpha=0.0)
        else:
            model = base_model
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_segs = (cfg.T - model_cfg.L) // model_cfg.stride + 1
        print(f"Params  {n_params:,}  K={model_cfg.K}  N~{n_segs} segments/window")

        if model_cfg.adversarial:
            trainer = AdversarialPipelineTrainer(
                model, model_cfg, trainer_cfg, device=device
            )
            header = f"\n{'Epoch':>6} | {'α':>7} | {'train_MSE':>10} | {'val_MSE':>10} | {'align':>10} | {'stab':>10} | {'lr':>10}"
            sep = "-" * 74
        else:
            trainer = PipelineTrainer(model, model_cfg, trainer_cfg, device=device)
            header = f"\n{'Epoch':>6} | {'train_MSE':>10} | {'val_MSE':>10} | {'align':>10} | {'lr':>10}"
            sep = "-" * 62
        trainer.checkpoint_path = ckpt_path

    print(header)
    print(sep)

    started_at = now_iso()
    t0 = time.time()
    trainer.fit(train_loader, val_loader, max_epochs=trainer_cfg.max_epochs)
    train_elapsed = time.time() - t0

    # ── Test evaluation (denormalised) ───────────────────────────────────
    model.load_state_dict(
        torch.load(trainer.checkpoint_path, map_location=device, weights_only=True)
    )
    val_m = eval_denorm(model, val_loader, mean, std, device)
    test_m = eval_denorm(model, test_loader, mean, std, device)

    total_elapsed = time.time() - t0

    # ── Persist results ──────────────────────────────────────────────────
    results = {
        "run_name": run_name,
        "model_type": cfg.model_type,
        "dataset": dataset_cfg.dataset,
        "T": cfg.T,
        "H": cfg.H,
        "n_params": n_params,
        "started_at": started_at,
        "finished_at": now_iso(),
        "train_elapsed_s": round(train_elapsed, 1),
        "total_elapsed_s": round(total_elapsed, 1),
        "elapsed_str": elapsed_str(total_elapsed),
        "best_val_mse_norm": round(trainer.best_val_mse, 6),
        "val_mse": val_m["mse"],
        "val_mae": val_m["mae"],
        "val_rmse": val_m["rmse"],
        "test_mse": test_m["mse"],
        "test_mae": test_m["mae"],
        "test_rmse": test_m["rmse"],
        "checkpoint_path": trainer.checkpoint_path,
    }
    results_path = save_results(results, run_name)

    print(f"\n{'─' * 40}")
    print(
        f"  val   MSE={val_m['mse']:.4f}  MAE={val_m['mae']:.4f}  RMSE={val_m['rmse']:.4f}"
    )
    print(
        f"  test  MSE={test_m['mse']:.4f}  MAE={test_m['mae']:.4f}  RMSE={test_m['rmse']:.4f}"
    )
    print(f"  time  {elapsed_str(total_elapsed)}  ({total_elapsed:.1f}s)")
    print(f"  ckpt  {trainer.checkpoint_path}")
    print(f"  json  {results_path}")
    return results
