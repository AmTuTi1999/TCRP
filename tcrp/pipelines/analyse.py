r"""TCRP analysis pipeline — relevance propagation pass on a saved checkpoint.

Supports both forecasting (TCRPForecaster) and classification (TCRPClassifier).

Forecasting:
    python -m tcrp.pipelines.analyse \
        --config configs/train.yaml \
        --checkpoint checkpoints/ETTh1_T336_H96_best.pt \
        --h-star 0 --n-samples 32

Classification:
    python -m tcrp.pipelines.analyse \
        --config configs/train_classification.yaml \
        +experiments/classification=exp_c01_ecg5000 \
        --checkpoint checkpoints/EXP-C01_seed42_best.pt \
        --k-star null --n-samples 32

Optional flags (both modes):
    --n-samples 32      Number of test samples to analyse (default: 32)
    --out path/to.pt    Save explanation tensors here (default: print only)
Forecasting only:
    --h-star 0          Forecast horizon step to explain (default: 0)
Classification only:
    --k-star <int>      Class index to explain for all samples (default: per-sample argmax)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

from tcrp.analysis.tcrp_analysis import TCRPAnalyser, verify_conservation
from tcrp.analysis.visualise import plot_explanation
from tcrp.dataset.datasets import DATASET_META
from tcrp.model.tcrp_forecaster.forecaster import TCRPConfig, TCRPForecaster
from tcrp.utils.misc import seed_everything

from .config import tcrp_config_from_hydra
from .train import build_loaders

# ── Data validation ──────────────────────────────────────────────────────────


def _check_data(cfg: DictConfig) -> None:
    """Raise FileNotFoundError with a clear message if required data is missing."""
    meta = DATASET_META.get(cfg.dataset)
    if meta is None:
        raise ValueError(f"Unknown dataset '{cfg.dataset}'")

    if "filename" in meta:
        # Forecasting dataset — verify the CSV exists.
        csv_path = Path(cfg.data_root) / meta["filename"]
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {csv_path}\n"
                f"Download {cfg.dataset} and place it at {csv_path}"
            )
    else:
        # Classification dataset — verify the data_root directory exists.
        root = Path(meta["data_root"])
        if not root.exists():
            raise FileNotFoundError(
                f"Classification dataset directory not found: {root}\n"
                f"See tcrp/data/classification_datasets.py for download instructions."
            )


# ── Checkpoint helpers ────────────────────────────────────────────────────────


def _load_state(path: str, device: torch.device) -> dict:
    """Load a state-dict, stripping adversarial 'base.' prefix if present."""
    state = torch.load(path, map_location=device, weights_only=True)
    if all(k.startswith("base.") for k in state):
        state = {k[len("base.") :]: v for k, v in state.items()}
    return state


def _find_checkpoint(cfg: DictConfig, explicit: str | None) -> str:
    """Resolve a checkpoint path.

    Priority:
        1. Explicit --checkpoint argument.
        2. Auto-discover: {checkpoint_dir}/{run_name}_best.pt using the same
           naming convention as the training pipeline.
    """
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {explicit}")
        return str(p)

    run_name = cfg.run_name or f"{cfg.datasets.dataset}_best"
    candidate = Path(cfg.checkpoint_dir) / f"{run_name}_best.pt"
    if candidate.exists():
        return str(candidate)

    raise FileNotFoundError(
        f"No checkpoint found. Tried: {candidate}\n"
        f"Pass --checkpoint <path> explicitly."
    )


# ── Shared plot helpers ───────────────────────────────────────────────────────


def _plot_samples(
    analyser: TCRPAnalyser,
    test_loader,
    concept_names: list[str],
    run_name: str,
    run_stamp: Path,
    n_plot: int,
    device: torch.device,
    h_star: int = 0,
    k_star: int | None = None,
    label: str | None = None,
    highlight_seg: int | None = None,
    is_cls: bool = False,
) -> None:
    test_ds = test_loader.dataset
    n_test = len(test_ds)
    plot_indices = np.linspace(0, n_test - 1, n_plot, dtype=int).tolist()
    print(f"Visualising {n_plot} test samples at dataset indices {plot_indices}")

    vis_x = torch.stack([test_ds[i][0] for i in plot_indices]).to(device)
    if is_cls:
        vis_expl = analyser.analyse(vis_x, k_star=k_star)
    else:
        vis_expl = analyser.analyse(vis_x, h_star=h_star)
    vis_x_cpu = vis_x.cpu()

    for vis_idx, ds_idx in enumerate(plot_indices):
        plot_explanation(
            vis_expl,
            vis_x_cpu,
            concept_names,
            run_id=run_name,
            h_star=h_star,
            out_dir=run_stamp / f"sample_{ds_idx}",
            sample_idx=vis_idx,
            highlight_seg=highlight_seg,
            label=label,
        )


# ── Main entry point ──────────────────────────────────────────────────────────


def analyse(cfg: DictConfig) -> dict:
    """Run the TCRP analysis pass and return a results dict.

    Dispatches to the forecasting or classification branch based on
    ``cfg.model_type``. Both branches share the same encoder/projection/pool
    LRP logic; only the decoder relevance initialisation differs.

    Returns:
        Dict with keys: run_name, n_samples, conserved, concept_names,
        figures_dir, explanation (dict of CPU tensors), and either h_star
        (forecasting) or k_star (classification).
    """
    seed_everything(cfg.seed)
    is_cls = cfg.get("model_type", "tcrp") == "tcrp_classifier"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_samples = cfg.get("n_samples", 32)
    n_plot = int(cfg.get("n_plot_samples", 1))
    hl_raw = cfg.get("highlight_seg", None)
    highlight_seg = int(hl_raw) if hl_raw is not None else None

    _now = datetime.now()

    if is_cls:
        return _analyse_cls(cfg, device, n_samples, n_plot, highlight_seg, _now)
    else:
        return _analyse_forecast(cfg, device, n_samples, n_plot, highlight_seg, _now)


# ── Forecasting branch ────────────────────────────────────────────────────────


def _analyse_forecast(
    cfg: DictConfig,
    device: torch.device,
    n_samples: int,
    n_plot: int,
    highlight_seg: int | None,
    _now: datetime,
) -> dict:
    dataset_cfg = cfg.datasets
    h_star = cfg.get("h_star", 0)

    _check_data(dataset_cfg)

    run_name = cfg.run_name or f"{dataset_cfg.dataset}_T{cfg.T}_H{cfg.H}"

    _, _, test_loader, _, _ = build_loaders(cfg)

    model_cfg: TCRPConfig = tcrp_config_from_hydra(cfg)
    model = TCRPForecaster(model_cfg).to(device)
    model.load_state_dict(_load_state(cfg.checkpoint_dir, device))
    model.eval()

    if h_star < 0 or h_star >= cfg.H:
        raise ValueError(f"h_star={h_star} out of range [0, {cfg.H})")

    x_batch, _ = next(iter(test_loader))
    x_batch = x_batch[:n_samples].to(device)

    analyser = TCRPAnalyser(model, eps=1e-6)
    explanation = analyser.analyse(x_batch, h_star=h_star)

    with torch.no_grad():
        out = model(x_batch)

    conserved = verify_conservation(explanation, out.y_hat, h_star=h_star, tol=1e-3)
    label = f"h*={h_star}"

    concept_names = model.scorer.concept_names
    _print_summary(
        run_name,
        cfg.checkpoint_dir,
        label,
        x_batch.shape[0],
        conserved,
        explanation.R_h,
        concept_names,
    )

    run_stamp = (
        Path(cfg.get("figures_dir", "figures"))
        / run_name
        / _now.strftime("%Y-%m-%d")
        / _now.strftime("%H-%M-%S")
    )
    _plot_samples(
        analyser,
        test_loader,
        concept_names,
        run_name,
        run_stamp,
        n_plot,
        device,
        h_star=h_star,
        label=label,
        highlight_seg=highlight_seg,
    )

    return {
        "run_name": run_name,
        "h_star": h_star,
        "n_samples": x_batch.shape[0],
        "conserved": conserved,
        "concept_names": concept_names,
        "figures_dir": str(run_stamp),
        "explanation": {
            "R_h": explanation.R_h.cpu(),
            "R_A": explanation.R_A.cpu(),
            "R_x": explanation.R_x.cpu(),
            "R_x_cond": explanation.R_x_cond.cpu(),
            "eta": explanation.eta.cpu(),
            "A": explanation.A.cpu(),
            "C": explanation.C.cpu(),
        },
    }


# ── Classification branch ─────────────────────────────────────────────────────


def _analyse_cls(
    cfg: DictConfig,
    device: torch.device,
    n_samples: int,
    n_plot: int,
    highlight_seg: int | None,
    _now: datetime,
) -> dict:
    from tcrp.data.classification_datasets import build_classification_loaders
    from tcrp.model.classifier import TCRPClassifier

    from .config import tcrp_class_config_from_hydra

    dataset_cfg = cfg.datasets
    _check_data(dataset_cfg)

    # k_star: None → per-sample argmax; int → explain that class for all samples
    k_star_raw = cfg.get("k_star", None)
    k_star: int | None = int(k_star_raw) if k_star_raw is not None else None

    run_name = (
        cfg.run_name or f"{dataset_cfg.dataset}_T{dataset_cfg.T}_C{dataset_cfg.C}"
    )

    loaders = build_classification_loaders(
        dataset_name=dataset_cfg.dataset,
        batch_size=cfg.trainers.get("batch_size", 64),
        data_root=(
            str(dataset_cfg.data_root) if hasattr(dataset_cfg, "data_root") else None
        ),
    )
    test_loader = loaders["test"]

    cls_config = tcrp_class_config_from_hydra(cfg)
    model = TCRPClassifier(cls_config).to(device)
    model.load_state_dict(_load_state(cfg.checkpoint_dir, device))
    model.eval()

    x_batch, y_batch = next(iter(test_loader))
    x_batch = x_batch[:n_samples].to(device)

    analyser = TCRPAnalyser(model, eps=1e-6)
    explanation = analyser.analyse(x_batch, k_star=k_star)

    with torch.no_grad():
        out = model(x_batch)

    # Per-sample targets for conservation check
    k_stars = explanation.k_stars  # (B,) — set by analyser
    target = out.y_hat.gather(1, k_stars.unsqueeze(1)).squeeze(1)
    conserved = verify_conservation(explanation, out.y_hat, target=target, tol=1e-3)

    k_display = k_star if k_star is not None else "pred"
    label = f"k*={k_display}"

    concept_names = model.scorer.concept_names
    _print_summary(
        run_name,
        cfg.checkpoint_dir,
        label,
        x_batch.shape[0],
        conserved,
        explanation.R_h,
        concept_names,
    )

    # Per-sample predicted classes summary
    preds = out.y_hat.argmax(dim=-1).cpu()
    true_labels = y_batch[:n_samples]
    acc = (preds == true_labels).float().mean().item()
    print(f"  accuracy on batch : {acc:.3f}")
    print(f"  predicted classes : {preds.tolist()}")
    print(f"  true classes      : {true_labels.tolist()}")
    print(f"{'=' * 60}\n")

    run_stamp = (
        Path(cfg.get("figures_dir", "figures"))
        / run_name
        / _now.strftime("%Y-%m-%d")
        / _now.strftime("%H-%M-%S")
    )
    _plot_samples(
        analyser,
        test_loader,
        concept_names,
        run_name,
        run_stamp,
        n_plot,
        device,
        k_star=k_star,
        label=label,
        highlight_seg=highlight_seg,
        is_cls=True,
    )

    return {
        "run_name": run_name,
        "k_star": k_star,
        "n_samples": x_batch.shape[0],
        "conserved": conserved,
        "concept_names": concept_names,
        "figures_dir": str(run_stamp),
        "explanation": {
            "R_h": explanation.R_h.cpu(),
            "R_A": explanation.R_A.cpu(),
            "R_x": explanation.R_x.cpu(),
            "R_x_cond": explanation.R_x_cond.cpu(),
            "eta": explanation.eta.cpu(),
            "A": explanation.A.cpu(),
            "C": explanation.C.cpu(),
            "k_stars": explanation.k_stars.cpu(),
        },
    }


# ── Shared print helper ───────────────────────────────────────────────────────


def _print_summary(
    run_name: str,
    checkpoint: str,
    label: str,
    n_samples: int,
    conserved: bool,
    R_h: torch.Tensor,
    concept_names: list[str],
) -> None:
    top_n = min(5, len(concept_names))
    R_h_mean = R_h.abs().mean(0)
    top_k = R_h_mean.topk(top_n)

    print(f"\n{'=' * 60}")
    print(f"  TCRP Analysis — {run_name}")
    print(f"  checkpoint  : {checkpoint}")
    print(f"  target      : {label}")
    print(f"  samples     : {n_samples}")
    print(f"  conservation: {'PASS' if conserved else 'FAIL (check tol)'}")
    print(f"\n  Top-{top_n} concept relevances (mean |R_h| over batch):")
    for rank, (idx, val) in enumerate(
        zip(top_k.indices.tolist(), top_k.values.tolist(), strict=False), start=1
    ):
        name = concept_names[idx] if idx < len(concept_names) else f"concept_{idx}"
        print(f"    {rank}. [{idx:2d}] {name:<32s}  {val:.6f}")
