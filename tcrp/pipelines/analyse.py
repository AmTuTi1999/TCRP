r"""TCRP analysis pipeline — relevance propagation pass on a saved checkpoint.

Run from the project root:
    python -m tcrp.pipelines.analyse \
        --config tcrp/pipelines/configs/etth1.yaml \
        --checkpoint checkpoints/ETTh1_T336_H96_best.pt

Optional flags:
    --h-star 0          Forecast horizon step to explain (default: 0 = first)
    --n-samples 32      Number of test samples to run (default: 32)
    --out path/to.pt    Save explanation tensors here (default: print only)
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


def _check_data(cfg: DictConfig) -> None:
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
    cfg: DictConfig,
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
    seed_everything(cfg.seed)

    dataset_cfg = cfg.datasets
    model_cfg = cfg.models

    h_star = cfg.h_star
    n_samples = cfg.n_samples

    _check_data(dataset_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = cfg.run_name or f"{dataset_cfg.dataset}_T{cfg.T}_H{cfg.H}"

    _, _, test_loader, _, _ = build_loaders(cfg)

    model_cfg: TCRPConfig = tcrp_config_from_hydra(cfg)
    model = TCRPForecaster(model_cfg).to(device)
    state = torch.load(cfg.checkpoint_dir, map_location=device, weights_only=True)
    # Adversarial checkpoints wrap the base model under a 'base.' prefix.
    # Strip it so analysis always runs on a plain TCRPForecaster.
    if all(k.startswith("base.") for k in state):
        state = {k[len("base.") :]: v for k, v in state.items()}
    model.load_state_dict(state)
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
    R_h_mean = explanation.R_h.abs().mean(0)  # (K,)
    top_n = min(5, len(concept_names))
    top_k = R_h_mean.topk(top_n)

    print(f"\n{'=' * 60}")
    print(f"  TCRP Analysis — {run_name}")
    print(f"  checkpoint : {cfg.checkpoint_dir}")
    print(f"  h_star     : {h_star}  (horizon step {h_star + 1}/{cfg.H})")
    print(f"  samples    : {x_batch.shape[0]}")
    print(f"  conservation: {'PASS' if conserved else 'FAIL (check tol)'}")
    print(f"\n  Top-{top_n} concept relevances (mean |R_h| over batch):")
    for rank, (idx, val) in enumerate(
        zip(top_k.indices.tolist(), top_k.values.tolist(), strict=False), start=1
    ):
        name = concept_names[idx] if idx < len(concept_names) else f"concept_{idx}"
        print(f"    {rank}. [{idx:2d}] {name:<32s}  {val:.6f}")
    print(f"{'=' * 60}\n")

    hl_raw = cfg.get("highlight_seg", None)
    highlight_seg = int(hl_raw) if hl_raw is not None else None

    _now = datetime.now()
    run_stamp = (
        Path(cfg.get("figures_dir", "figures"))
        / run_name
        / _now.strftime("%Y-%m-%d")
        / _now.strftime("%H-%M-%S")
    )

    n_plot = int(cfg.get("n_plot_samples", 1))
    test_ds = test_loader.dataset
    n_test = len(test_ds)
    plot_indices = np.linspace(0, n_test - 1, n_plot, dtype=int).tolist()

    print(f"Visualising {n_plot} test samples at dataset indices {plot_indices}")
    vis_x = torch.stack([test_ds[i][0] for i in plot_indices]).to(device)
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


def _find_checkpoint(cfg: DictConfig, explicit: str | None) -> str:
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
