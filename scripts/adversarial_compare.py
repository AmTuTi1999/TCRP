"""
Phase T*-06 · Adversarial vs Standard TCRP Comparison Experiment

Usage:
    python scripts/adversarial_compare.py --dataset ETTh1 --H 96 --seed 42

Trains two models on identical data and seeds:
  1. Standard TCRP (Trainer, adversarial=False)
  2. Adversarial TCRP (AdversarialTrainer, adversarial=True)

Reports for each:
  - Val MSE, MAE            (forecast quality)
  - Mean CAS per concept    (alignment quality)
  - Mean concept purity     (T*-05 contamination level)

T-29/T-31 diagnostics (bypass_ratio, concept_direction_gap) are imported
optionally — the script runs without them if those modules are not yet present.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tcrp.model.tcrp_forecaster.forecaster import TCRPConfig, TCRPForecaster
from tcrp.model.tcrp_forecaster.components.adversarial import AdversarialTCRPForecaster
from tcrp.training.trainer import Trainer
from tcrp.training.adversarial_trainer import AdversarialTrainer
from tcrp.dataset.datasets import get_loaders
from tcrp.diagnostics.concept_purity import concept_purity_report
from tcrp.utils import elapsed_str, seed_everything, compute_cas, gather_segments, save_results, now_iso

try:
    from tcrp.diagnostics.overfit import bypass_ratio, concept_direction_gap  # type: ignore
    _HAS_OVERFIT_DIAG = True
except ImportError:
    _HAS_OVERFIT_DIAG = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------



def make_config(args: argparse.Namespace, adversarial: bool) -> TCRPConfig:
    return TCRPConfig(
            H=args.H, d=64, K=0,
            L=20, stride=5,
            periods=[24, 168], k_max=2,
            alpha=5.0, beta=5.0, gamma=0.5, gamma_j=2.0,
            jump_threshold=3.0, train_std=1.0,
            lambda1=0.1, lambda2=1e-4, lambda3=0.01,
            probabilistic=False,
            adversarial=adversarial, alpha_max=args.alpha_max,
            warmup_epochs=args.warmup_epochs,
            encoder_in_ch=1, encoder_hidden=64,
            tcn_encoder_n_layers=4, tcn_encoder_kernel_size=3,
            tcn_encoder_use_weight_norm=True,
            lstm_encoder_bidirectional=False, lstm_encoder_dropout=0.0,
            lstm_encoder_pooling='last',
            attention_hidden=32, attention_temp=1.0,
        )


def purity_from_segments(
    model: AdversarialTCRPForecaster,
    xs: torch.Tensor,
    train_xs: torch.Tensor,
) -> dict:
    """Run concept_purity_report on first 256 val segments."""
    seg_fn = model.base.segmenter
    with torch.no_grad():
        val_segs_raw  = seg_fn(xs[:256]).reshape(-1, model.base.config.L)
        train_segs_raw = seg_fn(train_xs[:256]).reshape(-1, model.base.config.L)
    return concept_purity_report(model, model.base.scorer, train_segs_raw, val_segs_raw)


def run_experiment(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_started_at = now_iso()
    t_run = time.time()
    results: dict = {}

    train_loader, val_loader, _ = get_loaders(
        dataset=args.dataset,
        T=args.T,
        H=args.H,
        batch_size=args.batch_size,
    )

    # ------------------------------------------------------------------
    # 1. Standard TCRP
    # ------------------------------------------------------------------
    print("\n=== Standard TCRP ===")
    seed_everything(args.seed)
    cfg_std   = make_config(args, adversarial=False)
    model_std = TCRPForecaster(cfg_std).to(device)
    trainer_std = Trainer(model_std, cfg_std, device)

    std_started_at = now_iso()
    t_std = time.time()
    trainer_std.fit(train_loader, val_loader, max_epochs=args.epochs)
    std_elapsed = time.time() - t_std
    std_finished_at = now_iso()

    model_std.eval()
    val_mse_std = val_mae_std = 0.0
    n = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model_std(x)
            val_mse_std += torch.nn.functional.mse_loss(out.y_hat, y, reduction="sum").item()
            val_mae_std += torch.nn.functional.l1_loss(out.y_hat, y, reduction="sum").item()
            n += x.shape[0]
    val_mse_std /= n
    val_mae_std /= n
    cas_std = compute_cas(model_std, val_loader, device)

    results["standard"] = {
        "started_at":   std_started_at,
        "finished_at":  std_finished_at,
        "elapsed_s":    round(std_elapsed, 1),
        "elapsed_str":  elapsed_str(std_elapsed),
        "val_mse":      val_mse_std,
        "val_mae":      val_mae_std,
        "mean_cas":     cas_std,
    }

    # ------------------------------------------------------------------
    # 2. Adversarial TCRP
    # ------------------------------------------------------------------
    print("\n=== Adversarial TCRP ===")
    seed_everything(args.seed)
    cfg_adv    = make_config(args, adversarial=True)
    base_model = TCRPForecaster(cfg_adv).to(device)
    model_adv  = AdversarialTCRPForecaster(base_model, alpha=0.0)
    trainer_adv = AdversarialTrainer(model_adv, cfg_adv, device)

    adv_started_at = now_iso()
    t_adv = time.time()
    trainer_adv.fit(train_loader, val_loader, max_epochs=args.epochs)
    adv_elapsed = time.time() - t_adv
    adv_finished_at = now_iso()

    model_adv.eval()
    val_mse_adv = val_mae_adv = 0.0
    n = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out, _ = model_adv(x)
            val_mse_adv += torch.nn.functional.mse_loss(out.y_hat, y, reduction="sum").item()
            val_mae_adv += torch.nn.functional.l1_loss(out.y_hat, y, reduction="sum").item()
            n += x.shape[0]
    val_mse_adv /= n
    val_mae_adv /= n
    cas_adv = compute_cas(model_adv, val_loader, device)

    # Purity report
    train_xs = gather_segments(train_loader, device)
    val_xs   = gather_segments(val_loader,   device)
    purity   = purity_from_segments(model_adv, val_xs, train_xs)
    pure_count  = sum(1 for v in purity.values() if v["pure_val"])
    mean_purity = sum(v["cosine_val"] for v in purity.values()) / max(len(purity), 1)

    results["adversarial"] = {
        "started_at":     adv_started_at,
        "finished_at":    adv_finished_at,
        "elapsed_s":      round(adv_elapsed, 1),
        "elapsed_str":    elapsed_str(adv_elapsed),
        "val_mse":        val_mse_adv,
        "val_mae":        val_mae_adv,
        "mean_cas":       cas_adv,
        "mean_purity":    mean_purity,
        "pure_concepts":  pure_count,
        "total_concepts": len(purity),
        "purity_detail":  purity,
    }

    # Optional T-29/T-31 diagnostics
    if _HAS_OVERFIT_DIAG:
        results["adversarial"]["bypass_ratio"]    = bypass_ratio(model_adv, val_loader, device)
        results["adversarial"]["concept_dir_gap"] = concept_direction_gap(model_adv, train_loader, val_loader, device)
        results["standard"]["bypass_ratio"]       = bypass_ratio(model_std, val_loader, device)
        results["standard"]["concept_dir_gap"]    = concept_direction_gap(model_std, train_loader, val_loader, device)

    total_elapsed = time.time() - t_run
    results["meta"] = {
        "dataset":           args.dataset,
        "T":                 args.T,
        "H":                 args.H,
        "seed":              args.seed,
        "epochs":            args.epochs,
        "started_at":        run_started_at,
        "finished_at":       now_iso(),
        "total_elapsed_s":   round(total_elapsed, 1),
        "total_elapsed_str": elapsed_str(total_elapsed),
    }
    return results


def print_report(results: dict) -> None:
    print("\n" + "=" * 60)
    print("ADVERSARIAL vs STANDARD TCRP — COMPARISON REPORT")
    print("=" * 60)
    for name, r in results.items():
        print(f"\n  [{name.upper()}]")
        print(f"    Val MSE            : {r['val_mse']:.6f}")
        print(f"    Val MAE            : {r['val_mae']:.6f}")
        print(f"    Mean CAS           : {r['mean_cas']:.6f}")
        if "mean_purity" in r:
            print(f"    Mean purity (cos)  : {r['mean_purity']:.4f}")
            print(f"    Pure concepts      : {r['pure_concepts']}/{r['total_concepts']}")
        if "bypass_ratio" in r:
            print(f"    Bypass ratio       : {r['bypass_ratio']:.4f}")
        if "concept_dir_gap" in r:
            print(f"    Concept dir gap    : {r['concept_dir_gap']:.4f}")
    print()

    # Success criteria
    std = results.get("standard", {})
    adv = results.get("adversarial", {})
    print("  SUCCESS CRITERIA")
    mse_ok    = adv.get("val_mse", 1e9) <= std.get("val_mse", 0) * 1.05
    purity_ok = adv.get("mean_purity", 0) > 0.5
    cas_ok    = adv.get("mean_cas", 1e9) <= std.get("mean_cas", 0) * 1.05
    print(f"    MSE equal or lower (+5% tol)  : {'PASS' if mse_ok    else 'FAIL'}")
    print(f"    Mean purity > 0.5             : {'PASS' if purity_ok else 'FAIL'}")
    print(f"    CAS equal or better (+5% tol) : {'PASS' if cas_ok    else 'FAIL'}")

    if not mse_ok and purity_ok:
        mse_delta = (adv.get("val_mse", 0) - std.get("val_mse", 0)) / max(std.get("val_mse", 1), 1e-9)
        print(f"\n  NOTE: Pareto trade-off — adversarial raises MSE by {mse_delta*100:.1f}%"
              " while improving concept purity.  Consider accepting this trade-off"
              " if pure concept directions are required for the deployment context.")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Adversarial vs standard TCRP comparison")
    parser.add_argument("--dataset",       default="ETTh1")
    parser.add_argument("--H",             type=int, default=96)
    parser.add_argument("--T",             type=int, default=336)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--epochs",        type=int, default=50)
    parser.add_argument("--batch_size",    type=int, default=32)
    parser.add_argument("--alpha_max",     type=float, default=1.0)
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--out",           default=None, help="Optional JSON output path")
    args = parser.parse_args()

    results = run_experiment(args)
    print_report(results)

    run_name = f"adv_compare_{args.dataset}_H{args.H}"
    results_path = save_results(results, run_name, path=args.out or None)
    print(f"Results written to {results_path}")


if __name__ == "__main__":
    main()
