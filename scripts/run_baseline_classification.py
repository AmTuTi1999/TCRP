"""Baseline classification experiment runner.

Trains MLP, LSTM, FCN, and ResNet classifiers on the same datasets as the
TCRP classification experiments (EXP-C01 … EXP-C08) and saves metrics in the
same format as run_classification.py for direct comparison.

Usage:
    # Single model + experiment
    python scripts/run_baseline_classification.py --experiment EXP-C01 --model fcn

    # All models for one experiment
    python scripts/run_baseline_classification.py --experiment EXP-C01 --model all

    # All models × all experiments
    python scripts/run_baseline_classification.py --model all
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tcrp.dataset.classification_datasets import build_classification_loaders
from tcrp.eval.classification_metrics import evaluate_all
from tcrp.model.classification_baselines import (
    BASELINE_MODELS,
    build_baseline_classifier,
)
from tcrp.utils.misc import seed_everything

# ── Experiment registry (mirrors run_classification.py) ──────────────────────

EXPERIMENTS: dict[str, dict] = {
    "EXP-C01": {
        "dataset": "ECG5000",
        "C": 5,
        "trainer": {"batch_size": 64, "max_epochs": 150, "es_patience": 20, "lr": 1e-3},
        "weighted_sampling": False,
    },
    "EXP-C02": {
        "dataset": "MITBIH",
        "C": 5,
        "trainer": {
            "batch_size": 128,
            "max_epochs": 150,
            "es_patience": 20,
            "lr": 1e-3,
        },
        "weighted_sampling": True,
    },
    "EXP-C03": {
        "dataset": "SleepEDF",
        "C": 5,
        "trainer": {"batch_size": 32, "max_epochs": 150, "es_patience": 20, "lr": 5e-4},
        "weighted_sampling": True,
    },
    "EXP-C04": {
        "dataset": "CWRU",
        "C": 4,
        "trainer": {"batch_size": 64, "max_epochs": 150, "es_patience": 20, "lr": 1e-3},
        "weighted_sampling": False,
    },
    "EXP-C05": {
        "dataset": "UCIHAR",
        "C": 6,
        "trainer": {"batch_size": 64, "max_epochs": 150, "es_patience": 20, "lr": 1e-3},
        "weighted_sampling": False,
    },
    "EXP-C06": {
        "dataset": "Ethanol",
        "C": 4,
        "trainer": {"batch_size": 32, "max_epochs": 150, "es_patience": 20, "lr": 5e-4},
        "weighted_sampling": False,
    },
    "EXP-C07-A": {
        "dataset": "SP500_A",
        "C": 2,
        "trainer": {"batch_size": 32, "max_epochs": 150, "es_patience": 20, "lr": 5e-4},
        "weighted_sampling": True,
    },
    "EXP-C07-B": {
        "dataset": "SP500_B",
        "C": 3,
        "trainer": {"batch_size": 32, "max_epochs": 150, "es_patience": 20, "lr": 5e-4},
        "weighted_sampling": False,
    },
    "EXP-C08": {
        "dataset": "FX",
        "C": 3,
        "trainer": {"batch_size": 64, "max_epochs": 200, "es_patience": 30, "lr": 1e-3},
        "weighted_sampling": False,
    },
}


class _BaselineCETrainer:
    """Minimal cross-entropy trainer for baseline classifiers."""

    def __init__(
        self,
        model: nn.Module,
        lr: float,
        es_patience: int,
        checkpoint_path: str,
        class_weights: torch.Tensor | None,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.class_weights = class_weights
        self.checkpoint_path = checkpoint_path
        self._es_patience = es_patience

        self.optimizer = Adam(model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5
        )
        self.best_val_ce = float("inf")
        self.epochs_no_improve = 0

    def _ce(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        weights = (
            self.class_weights.to(self.device)
            if self.class_weights is not None
            else None
        )
        return F.cross_entropy(logits, y, weight=weights)

    def _train_epoch(self, loader) -> float:
        self.model.train()
        total, count = 0.0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            loss = self._ce(self.model(x).y_hat, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total += loss.item() * x.shape[0]
            count += x.shape[0]
        return total / count

    @torch.no_grad()
    def _validate(self, loader) -> dict:
        self.model.eval()
        total_ce, correct, count = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x).y_hat
            total_ce += F.cross_entropy(logits, y, reduction="sum").item()
            correct += (logits.argmax(dim=-1) == y).sum().item()
            count += x.shape[0]
        return {"ce": total_ce / count, "accuracy": correct / count}

    def fit(self, train_loader, val_loader, max_epochs: int) -> None:
        Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        print(
            f"\n{'Epoch':>6} | {'train_CE':>10} | {'val_CE':>10} | {'val_acc':>8} | {'lr':>10}"
        )
        print("-" * 56)

        for epoch in range(1, max_epochs + 1):
            train_ce = self._train_epoch(train_loader)
            val_m = self._validate(val_loader)
            self.scheduler.step(val_m["ce"])
            lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"{epoch:6d} | {train_ce:10.6f} | "
                f"{val_m['ce']:10.6f} | {val_m['accuracy']:8.4f} | {lr:10.2e}"
            )

            if val_m["ce"] < self.best_val_ce:
                self.best_val_ce = val_m["ce"]
                self.epochs_no_improve = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= self._es_patience:
                print(f"Early stopping at epoch {epoch}.")
                break


def run(
    experiment: str,
    model_type: str,
    seed: int = 42,
    results_dir: str = "results",
    checkpoint_dir: str = "checkpoints",
) -> dict:
    """Train and evaluate a single baseline on one experiment.

    Returns:
        Dict of evaluation metrics (same schema as run_classification.py).
    """
    if experiment not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment '{experiment}'. Available: {list(EXPERIMENTS)}"
        )
    if model_type not in BASELINE_MODELS:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Available: {list(BASELINE_MODELS)}"
        )

    cfg = EXPERIMENTS[experiment]
    seed_everything(seed)

    run_name = f"BL-{model_type.upper()}_{experiment}_seed{seed}"
    print(f"\n{'=' * 66}")
    print(f"  {run_name}")
    print(f"  dataset={cfg['dataset']}  model={model_type}  seed={seed}")
    print(f"{'=' * 66}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device={device}")

    # ── Data ────────────────────────────────────────────────────────────────
    loaders = build_classification_loaders(
        dataset_name=cfg["dataset"],
        batch_size=cfg["trainer"]["batch_size"],
        weighted_sampling=cfg["weighted_sampling"],
    )
    print(
        f"  train={loaders['n_train']:,}  val={loaders['n_val']:,}  "
        f"test={loaders['n_test']:,}"
    )

    # Infer T from first batch
    sample_x, _ = next(iter(loaders["train"]))
    T = sample_x.shape[1]
    C = cfg["C"]
    print(f"  T={T}  C={C}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_baseline_classifier(model_type, T=T, C=C)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params={n_params:,}")

    ckpt_path = str(Path(checkpoint_dir) / f"{run_name}_best.pt")

    # ── Train ────────────────────────────────────────────────────────────────
    trainer_kw = cfg["trainer"]
    trainer = _BaselineCETrainer(
        model=model,
        lr=trainer_kw["lr"],
        es_patience=trainer_kw["es_patience"],
        checkpoint_path=ckpt_path,
        class_weights=loaders["class_weights"] if cfg["weighted_sampling"] else None,
        device=device,
    )

    t0 = time.time()
    trainer.fit(loaders["train"], loaders["val"], max_epochs=trainer_kw["max_epochs"])
    train_elapsed = time.time() - t0

    # ── Evaluate ─────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    metrics = evaluate_all(
        model,
        loaders["test"],
        device=device,
        class_weights=loaders["class_weights"],
    )
    total_elapsed = time.time() - t0

    print(f"\n{'─' * 40}")
    print(f"  accuracy  {metrics['accuracy']:.4f}")
    print(f"  macro_F1  {metrics['macro_f1']:.4f}")
    print(f"  CE        {metrics['cross_entropy']:.4f}")
    print(f"  elapsed   {total_elapsed:.1f}s")
    print(f"  ckpt      {ckpt_path}")

    out_dir = Path(results_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_out = {
        "run_name": run_name,
        "experiment": experiment,
        "model_type": model_type,
        "dataset": cfg["dataset"],
        "seed": seed,
        "n_params": n_params,
        "T": T,
        "C": C,
        "train_elapsed_s": round(train_elapsed, 1),
        "total_elapsed_s": round(total_elapsed, 1),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "cross_entropy": metrics["cross_entropy"],
        "per_class_accuracy": {
            str(k): v for k, v in metrics["per_class_accuracy"].items()
        },
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "n_test": metrics["n_samples"],
        "checkpoint_path": ckpt_path,
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"  metrics   {out_dir / 'metrics.json'}")
    return metrics_out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline classification experiment runner"
    )
    parser.add_argument(
        "--experiment",
        choices=list(EXPERIMENTS) + ["all"],
        default="all",
        help="Experiment ID (e.g. EXP-C01) or 'all' to run every experiment",
    )
    parser.add_argument(
        "--model",
        choices=list(BASELINE_MODELS) + ["all"],
        default="all",
        help="Baseline model type or 'all' to run every model",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    experiments = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    models = list(BASELINE_MODELS) if args.model == "all" else [args.model]

    all_results = []
    for exp in experiments:
        for mdl in models:
            result = run(
                experiment=exp,
                model_type=mdl,
                seed=args.seed,
                results_dir=args.results_dir,
                checkpoint_dir=args.checkpoint_dir,
            )
            all_results.append(result)

    if len(all_results) > 1:
        print(f"\n{'=' * 66}")
        print(f"  Summary ({len(all_results)} runs)")
        print(f"  {'run_name':<40} {'acc':>6}  {'F1':>6}")
        print(f"  {'-' * 56}")
        for r in all_results:
            print(f"  {r['run_name']:<40} {r['accuracy']:6.4f}  {r['macro_f1']:6.4f}")
