"""Classification experiment runner for TCRP (TC-05).

Usage:
    python scripts/run_classification.py --experiment EXP-C01 --seed 42
    python scripts/run_classification.py --experiment EXP-C08 --seed 0 --adversarial

Maps experiment IDs (EXP-C01 … EXP-C08) to dataset/config, trains
TCRPClassifier, evaluates on test set, and saves:
  results/<run_name>/metrics.json        — accuracy, F1, CE, confusion matrix
  results/<run_name>/concept_profiles.json — per-class concept relevance
  checkpoints/<run_name>_best.pt         — best model weights
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from tcrp.dataset.classification_datasets import build_classification_loaders
from tcrp.eval.classification_metrics import evaluate_all
from tcrp.eval.concept_class_profile import class_concept_profiles
from tcrp.model.classifier import TCRPClassConfig, TCRPClassifier
from tcrp.model.tcrp_forecaster.components.adversarial import AdversarialTCRPClassifier
from tcrp.training.classification_trainer import (
    AdversarialClassificationTrainer,
    ClassificationTrainer,
)
from tcrp.utils.misc import seed_everything

# ── Experiment registry ──────────────────────────────────────────────────────

# Maps EXP-ID → (dataset_name, TCRPClassConfig kwargs, trainer kwargs)
EXPERIMENTS: dict[str, dict] = {
    "EXP-C01": {
        "dataset": "ECG5000",
        "model": {"C": 5, "L": 20, "stride": 5, "periods": [4, 6, 8, 10], "k_max": 2},
        "trainer": {"batch_size": 64, "max_epochs": 150, "es_patience": 20, "lr": 1e-3},
        "weighted_sampling": False,
    },
    "EXP-C02": {
        "dataset": "MITBIH",
        "model": {"C": 5, "L": 25, "stride": 5, "periods": [18, 36], "k_max": 2},
        "trainer": {
            "batch_size": 128,
            "max_epochs": 150,
            "es_patience": 20,
            "lr": 1e-2,
        },
        "weighted_sampling": True,
    },
    "EXP-C03": {
        "dataset": "SleepEDF",
        "model": {"C": 5, "L": 100, "stride": 50, "periods": [100, 500], "k_max": 2},
        "trainer": {"batch_size": 32, "max_epochs": 150, "es_patience": 20, "lr": 5e-4},
        "weighted_sampling": True,
    },
    "EXP-C04": {
        "dataset": "CWRU",
        "model": {"C": 4, "L": 64, "stride": 16, "periods": [32, 64], "k_max": 2},
        "trainer": {"batch_size": 64, "max_epochs": 150, "es_patience": 20, "lr": 1e-3},
        "weighted_sampling": False,
    },
    "EXP-C05": {
        "dataset": "UCIHAR",
        "model": {"C": 6, "L": 20, "stride": 5, "periods": [10, 20], "k_max": 2},
        "trainer": {"batch_size": 64, "max_epochs": 150, "es_patience": 20, "lr": 1e-3},
        "weighted_sampling": False,
    },
    "EXP-C06": {
        "dataset": "Ethanol",
        "model": {"C": 4, "L": 50, "stride": 20, "periods": [], "k_max": 2},
        "trainer": {"batch_size": 32, "max_epochs": 150, "es_patience": 20, "lr": 5e-4},
        "weighted_sampling": False,
    },
    "EXP-C07-A": {
        "dataset": "SP500_A",
        "model": {"C": 2, "L": 21, "stride": 5, "periods": [], "k_max": 2},
        "trainer": {"batch_size": 32, "max_epochs": 150, "es_patience": 20, "lr": 5e-4},
        "weighted_sampling": True,
    },
    "EXP-C07-B": {
        "dataset": "SP500_B",
        "model": {"C": 3, "L": 21, "stride": 5, "periods": [], "k_max": 2},
        "trainer": {"batch_size": 32, "max_epochs": 150, "es_patience": 20, "lr": 5e-4},
        "weighted_sampling": False,
    },
    "EXP-C08": {
        "dataset": "FX",
        "model": {
            "C": 3,
            "L": 6,
            "stride": 1,
            "periods": [],
            "k_max": 2,
            "tcn_encoder_n_layers": 2,
        },
        "trainer": {"batch_size": 64, "max_epochs": 200, "es_patience": 30, "lr": 1e-3},
        "weighted_sampling": False,
    },
}

# Concept names in the order the concept scorer produces them (K=18 base + periods)
# Indices 0-15: base concepts; 16-17: ACF lags (k_max=2); 18+: period concepts
BASE_CONCEPT_NAMES = [
    "trend_slope",  # 0
    "trend_r2",  # 1
    "monotonicity_up",  # 2
    "monotonicity_down",  # 3
    "curvature_pos",  # 4
    "curvature_neg",  # 5
    "volatility",  # 6
    "volatility_ratio",  # 7
    "stochasticity_xi",  # 8
    "hurst",  # 9
    "skewness",  # 10
    "kurtosis",  # 11
    "jump_indicator",  # 12
    "break_score",  # 13
    "tendency",  # 14
    "autocorr_lag1",  # 15
    "acf_k1",  # 16
    "acf_k2",  # 17
]


def concept_names_for(periods: list[int]) -> list[str]:
    """Build the full concept name list for a given period config."""
    return BASE_CONCEPT_NAMES + [f"period_{p}" for p in periods]


# ── Main runner ──────────────────────────────────────────────────────────────


def run(
    experiment: str,
    seed: int = 42,
    adversarial: bool = False,
    results_dir: str = "results",
    checkpoint_dir: str = "checkpoints",
) -> dict:
    """Train and evaluate a single classification experiment.

    Args:
        experiment: Experiment ID, e.g. 'EXP-C01'.
        seed: Random seed.
        adversarial: Enable adversarial (GRL) training.
        results_dir: Directory for metric JSON outputs.
        checkpoint_dir: Directory for model checkpoints.

    Returns:
        Dict of evaluation metrics.
    """
    if experiment not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment '{experiment}'. Available: {list(EXPERIMENTS)}"
        )

    cfg = EXPERIMENTS[experiment]
    seed_everything(seed)

    run_name = f"{experiment}_seed{seed}" + ("_adv" if adversarial else "")
    print(f"\n{'=' * 66}")
    print(f"  {run_name}")
    print(f"  dataset={cfg['dataset']}  seed={seed}  adversarial={adversarial}")
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

    # ── Model ────────────────────────────────────────────────────────────────
    model_kwargs = dict(cfg["model"])

    config = TCRPClassConfig(**model_kwargs)
    base_model = TCRPClassifier(config)

    if adversarial:
        model = AdversarialTCRPClassifier(base_model=base_model, alpha=0.0)
    else:
        model = base_model
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params={n_params:,}  K={config.K}  C={config.C}")

    ckpt_path = str(Path(checkpoint_dir) / f"{run_name}_best.pt")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ── Trainer ─────────────────────────────────────────────────────────────
    trainer_kw = cfg["trainer"]
    trainer_cls = (
        AdversarialClassificationTrainer if adversarial else ClassificationTrainer
    )
    trainer = trainer_cls(
        model=model,
        config=config,
        lr=trainer_kw["lr"],
        es_patience=trainer_kw["es_patience"],
        checkpoint_path=ckpt_path,
        class_weights=loaders["class_weights"] if cfg["weighted_sampling"] else None,
        device=device,
    )

    t0 = time.time()
    trainer.fit(
        loaders["train"],
        loaders["val"],
        max_epochs=trainer_kw["max_epochs"],
    )
    train_elapsed = time.time() - t0

    # ── Test evaluation ──────────────────────────────────────────────────────
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    # Unwrap adversarial wrapper — eval functions expect TCRPOutput, not a tuple
    eval_model = model.base if adversarial else model
    metrics = evaluate_all(
        eval_model,
        loaders["test"],
        device=device,
        class_weights=loaders["class_weights"],
    )

    # ── Concept profiles ─────────────────────────────────────────────────────
    c_names = concept_names_for(config.periods)
    profiles = class_concept_profiles(
        eval_model, loaders["test"], concept_names=c_names, device=device
    )

    total_elapsed = time.time() - t0
    print(f"\n{'─' * 40}")
    print(f"  accuracy  {metrics['accuracy']:.4f}")
    print(f"  macro_F1  {metrics['macro_f1']:.4f}")
    print(f"  CE        {metrics['cross_entropy']:.4f}")
    print(f"  elapsed   {total_elapsed:.1f}s")
    print(f"  ckpt      {ckpt_path}")

    # ── Persist ──────────────────────────────────────────────────────────────
    out_dir = Path(results_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_out = {
        "run_name": run_name,
        "experiment": experiment,
        "dataset": cfg["dataset"],
        "seed": seed,
        "adversarial": adversarial,
        "n_params": n_params,
        "K": config.K,
        "C": config.C,
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

    profiles_serialisable = {
        str(cls): {name: float(v) for name, v in profile.items()}
        for cls, profile in profiles.items()
    }
    with open(out_dir / "concept_profiles.json", "w") as f:
        json.dump(profiles_serialisable, f, indent=2)

    print(f"  metrics   {out_dir / 'metrics.json'}")
    print(f"  profiles  {out_dir / 'concept_profiles.json'}")

    return metrics_out


# ── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TCRP classification experiment runner"
    )
    parser.add_argument(
        "--experiment",
        required=True,
        choices=list(EXPERIMENTS),
        help="Experiment ID, e.g. EXP-C01",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--adversarial",
        action="store_true",
        default=False,
        help="Enable adversarial (GRL) training",
    )
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        experiment=args.experiment,
        seed=args.seed,
        adversarial=args.adversarial,
        results_dir=args.results_dir,
        checkpoint_dir=args.checkpoint_dir,
    )
