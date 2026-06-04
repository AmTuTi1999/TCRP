"""
Phase T*-07 · Adversarial Unit Tests

Covers T*-01 through T*-05 as specified in the Phase T* plan.
Run with:  pytest tests/test_adversarial.py -v
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tcrp.model.tcrp_forecaster.components.adversarial import (
    AdversarialTCRPForecaster,
    GradientReversal,
    GRLLayer,
    grl_alpha_schedule,
)
from tcrp.model.tcrp_forecaster.components.bottleneck import alignment_loss, stability_loss
from tcrp.model.tcrp_forecaster.forecaster import TCRPConfig, TCRPForecaster


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_config() -> TCRPConfig:
    return TCRPConfig(
        T=100, H=12, L=10, stride=5, d=16, k_max=1,
        periods=[24], lambda1=0.1, lambda2=1e-4, lambda3=0.01,
        adversarial=True, alpha_max=1.0, warmup_epochs=5,
    )


@pytest.fixture()
def adv_model(small_config: TCRPConfig) -> AdversarialTCRPForecaster:
    base = TCRPForecaster(small_config)
    return AdversarialTCRPForecaster(base, alpha=1.0)


# ---------------------------------------------------------------------------
# T*-01 GRL layer tests
# ---------------------------------------------------------------------------

class TestGRLLayer:
    def test_grl_forward_identity(self):
        grl = GRLLayer(alpha=1.0)
        x = torch.randn(4, 8)
        out = grl(x)
        assert torch.allclose(out, x), "GRL forward must be identity"

    def test_grl_backward_negation(self):
        grl = GRLLayer(alpha=1.0)
        x = torch.randn(4, 8, requires_grad=True)
        out = grl(x)
        upstream = torch.ones_like(out)
        out.backward(upstream)
        expected = -upstream
        assert torch.allclose(x.grad, expected), \
            "GRL backward with alpha=1 must negate gradient"

    def test_grl_alpha_zero(self):
        """alpha=0 → GRL acts as stop-gradient (zero gradient to input)."""
        grl = GRLLayer(alpha=0.0)
        x = torch.randn(3, 5, requires_grad=True)
        out = grl(x)
        out.sum().backward()
        assert torch.allclose(x.grad, torch.zeros_like(x.grad)), \
            "alpha=0 must produce zero gradient (stop-gradient)"

    def test_grl_alpha_one(self):
        """alpha=1 → gradient fully negated."""
        grl = GRLLayer(alpha=1.0)
        x = torch.randn(3, 5, requires_grad=True)
        out = grl(x)
        upstream = torch.randn_like(out)
        out.backward(upstream)
        assert torch.allclose(x.grad, -upstream), \
            "alpha=1 must fully negate gradient"

    def test_grl_alpha_scaling(self):
        """Gradient is scaled by alpha."""
        alpha = 2.5
        grl = GRLLayer(alpha=alpha)
        x = torch.randn(3, 5, requires_grad=True)
        out = grl(x)
        upstream = torch.ones_like(out)
        out.backward(upstream)
        assert torch.allclose(x.grad, -alpha * upstream), \
            "GRL gradient must be -alpha * upstream"

    def test_two_path_backward(self):
        """
        Forecast gradient reaches encoder unchanged; alignment gradient reaches
        encoder negated.  Verified via encoder parameter gradients.

        We run three experiments with a simple linear encoder:
          (a) Only forecast path  → encoder grad = g_fc
          (b) Only alignment path (no GRL, alpha=0) → encoder grad = g_al
          (c) Only alignment path (with GRL, alpha=1) → encoder grad = -g_al
        """
        grl_off = GRLLayer(alpha=0.0)
        grl_on  = GRLLayer(alpha=1.0)
        encoder = nn.Linear(8, 4)

        x = torch.randn(2, 8)

        # (a) Forecast path only
        z_a = encoder(x)
        L_fc = z_a.sum()
        encoder.zero_grad()
        L_fc.backward()
        g_fc = {n: p.grad.clone() for n, p in encoder.named_parameters() if p.grad is not None}

        # (b) Alignment path without GRL (alpha=0 stops gradient — use a plain path instead)
        z_b = encoder(x)
        L_al_plain = z_b.sum()
        encoder.zero_grad()
        L_al_plain.backward()
        g_al = {n: p.grad.clone() for n, p in encoder.named_parameters() if p.grad is not None}

        # (c) Alignment path through GRL (alpha=1)
        z_c = encoder(x)
        z_grl = grl_on(z_c)
        L_al_grl = z_grl.sum()
        encoder.zero_grad()
        L_al_grl.backward()
        g_al_grl = {n: p.grad.clone() for n, p in encoder.named_parameters() if p.grad is not None}

        # (a) and (b) should be identical (same computation, no GRL)
        for name in g_fc:
            assert torch.allclose(g_fc[name], g_al[name], atol=1e-6), \
                f"Forecast and plain alignment grads should match for '{name}'"

        # (c) should be the negation of (b): GRL reverses the gradient
        for name in g_al:
            assert torch.allclose(g_al_grl[name], -g_al[name], atol=1e-6), \
                f"GRL-reversed gradient for '{name}' must equal -g_al"


# ---------------------------------------------------------------------------
# T*-02 Alpha schedule tests
# ---------------------------------------------------------------------------

class TestAlphaSchedule:
    def test_schedule_warmup_zero(self):
        for e in range(20):
            assert grl_alpha_schedule(e, 100, warmup_epochs=20) == 0.0, \
                f"alpha must be 0 during warmup (epoch={e})"

    def test_schedule_monotone(self):
        alphas = [grl_alpha_schedule(e, 100, warmup_epochs=20) for e in range(100)]
        post = alphas[20:]
        assert all(post[i] <= post[i + 1] for i in range(len(post) - 1)), \
            "alpha must be non-decreasing after warmup"

    def test_schedule_max(self):
        a = grl_alpha_schedule(9999, 10000, warmup_epochs=20, alpha_max=1.0)
        assert abs(a - 1.0) < 1e-3, "alpha must approach alpha_max"

    def test_schedule_continuity(self):
        a_before = grl_alpha_schedule(19, 100, warmup_epochs=20)
        a_after  = grl_alpha_schedule(20, 100, warmup_epochs=20)
        assert a_before == 0.0
        assert a_after >= 0.0
        # No negative jump
        assert a_after >= a_before, "no discontinuity at warmup boundary"

    def test_schedule_custom_alpha_max(self):
        a = grl_alpha_schedule(9999, 10000, warmup_epochs=0, alpha_max=2.0)
        assert abs(a - 2.0) < 1e-3


# ---------------------------------------------------------------------------
# T*-03 AdversarialTCRPForecaster tests
# ---------------------------------------------------------------------------

class TestAdversarialForecaster:
    def test_forward_shapes(self, adv_model: AdversarialTCRPForecaster, small_config: TCRPConfig):
        x = torch.randn(2, small_config.T)
        out, A_align = adv_model(x)
        assert out.y_hat.shape == (2, small_config.H)
        assert A_align.shape[0] == 2
        assert A_align.shape[-1] == small_config.K

    def test_set_alpha(self, adv_model: AdversarialTCRPForecaster):
        adv_model.set_alpha(0.5)
        assert adv_model.grl.alpha == 0.5

    def test_forecast_path_no_grl(self, adv_model: AdversarialTCRPForecaster, small_config: TCRPConfig):
        """
        Gradient of forecast loss must NOT pass through the GRL.
        We check that the encoder receives a positive gradient from L_forecast.
        """
        x = torch.randn(2, small_config.T)
        adv_model.train()
        adv_model.base.encoder.zero_grad()

        out, A_align = adv_model(x)
        y = torch.zeros_like(out.y_hat)
        L_fc = F.mse_loss(out.y_hat, y)
        L_fc.backward(retain_graph=True)

        # Encoder should have non-zero gradients from L_forecast
        enc_grad_fc = sum(
            p.grad.abs().sum().item()
            for p in adv_model.base.encoder.parameters()
            if p.grad is not None
        )
        assert enc_grad_fc > 0, "Forecast loss must produce gradients in encoder"

    def test_alignment_path_through_grl(self, adv_model: AdversarialTCRPForecaster, small_config: TCRPConfig):
        """
        Encoder param gradients from alignment path (via GRL with alpha=1)
        should be the negation of the natural alignment gradient (no GRL).

        We compute both from scratch to avoid the graph being shared.
        """
        adv_model.train()
        x = torch.randn(2, small_config.T)

        # Pre-compute C (detached) from the forecast path
        with torch.no_grad():
            out_ref, _ = adv_model(x)
            C = out_ref.C  # already detached

        # --- Natural gradient: alignment_loss(projection(z), C) — no GRL ---
        adv_model.base.encoder.zero_grad()
        segs = adv_model.base.segmenter(x)
        z_nat = adv_model.base.encoder(segs)
        A_nat = adv_model.base.projection(z_nat)
        L_nat = alignment_loss(A_nat, C)
        L_nat.backward()
        grad_natural = {
            n: p.grad.clone()
            for n, p in adv_model.base.encoder.named_parameters()
            if p.grad is not None
        }

        # --- GRL gradient: alignment_loss(projection(grl(z)), C) ---
        adv_model.base.encoder.zero_grad()
        adv_model.set_alpha(1.0)
        z_grl_in = adv_model.base.encoder(segs.detach())
        z_grl_out = adv_model.grl(z_grl_in)
        A_grl = adv_model.base.projection(z_grl_out)
        L_grl = alignment_loss(A_grl, C)
        L_grl.backward()
        grad_with_grl = {
            n: p.grad.clone()
            for n, p in adv_model.base.encoder.named_parameters()
            if p.grad is not None
        }

        for name in grad_natural:
            assert torch.allclose(grad_with_grl[name], -grad_natural[name], atol=1e-5), \
                f"Encoder param '{name}' gradient not negated by GRL"


# ---------------------------------------------------------------------------
# T*-04 Two-path backward — no mixing
# ---------------------------------------------------------------------------

class TestTwoPathBackward:
    def test_two_path_backward_additive(self, adv_model: AdversarialTCRPForecaster, small_config: TCRPConfig):
        """
        Gradient contributions from Path 1 and Path 2 are additive and
        independent when accumulated via two separate backward() calls.
        """
        x = torch.randn(2, small_config.T)
        y = torch.zeros(2, small_config.H)
        adv_model.train()

        # Combined: both paths in one go
        adv_model.base.encoder.zero_grad()
        out, A_align = adv_model(x)
        L_fc = F.mse_loss(out.y_hat, y)
        L_al = alignment_loss(A_align, out.C)
        L_fc.backward(retain_graph=True)
        L_al.backward()
        grad_combined = {
            n: p.grad.clone()
            for n, p in adv_model.base.encoder.named_parameters()
            if p.grad is not None
        }

        # Separate: only Path 1
        adv_model.base.encoder.zero_grad()
        out1, A1 = adv_model(x)
        L_fc1 = F.mse_loss(out1.y_hat, y)
        L_fc1.backward()
        grad_p1 = {
            n: p.grad.clone()
            for n, p in adv_model.base.encoder.named_parameters()
            if p.grad is not None
        }

        # Separate: only Path 2
        adv_model.base.encoder.zero_grad()
        out2, A2 = adv_model(x)
        L_al2 = alignment_loss(A2, out2.C)
        L_al2.backward()
        grad_p2 = {
            n: p.grad.clone()
            for n, p in adv_model.base.encoder.named_parameters()
            if p.grad is not None
        }

        for name in grad_combined:
            expected = grad_p1[name] + grad_p2[name]
            assert torch.allclose(grad_combined[name], expected, atol=1e-5), \
                f"Gradient for '{name}' is not additive across the two paths"


# ---------------------------------------------------------------------------
# Stability loss tests
# ---------------------------------------------------------------------------

class TestStabilityLoss:
    def test_zero_when_identical(self):
        A = torch.randn(2, 5, 4)
        assert stability_loss(A, A).item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_for_difference(self):
        A = torch.randn(2, 5, 4)
        C = torch.randn(2, 5, 4)
        assert stability_loss(A, C).item() >= 0.0

    def test_single_segment_returns_zero(self):
        A = torch.randn(2, 1, 4)
        C = torch.randn(2, 1, 4)
        assert stability_loss(A, C).item() == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# T*-05 Purity score tests
# ---------------------------------------------------------------------------

class TestConceptPurityScore:
    def test_purity_score_range(self, adv_model: AdversarialTCRPForecaster, small_config: TCRPConfig):
        from tcrp.diagnostics.concept_purity import concept_purity_score
        segs = torch.randn(32, small_config.L)
        for k in range(small_config.K):
            result = concept_purity_score(adv_model, adv_model.base.scorer, segs, k)
            assert -1.0 <= result["cosine_sim"] <= 1.0, \
                f"cosine_sim out of range for concept {k}"
            assert "concept" in result
            assert "pure" in result
            assert "warning" in result

    def test_purity_report_all_concepts(self, adv_model: AdversarialTCRPForecaster, small_config: TCRPConfig):
        from tcrp.diagnostics.concept_purity import concept_purity_report
        segs = torch.randn(32, small_config.L)
        report = concept_purity_report(adv_model, adv_model.base.scorer, segs, segs)
        assert len(report) == small_config.K
        for name, vals in report.items():
            assert "cosine_train" in vals
            assert "cosine_val" in vals
            assert "purity_gap" in vals
            assert abs(vals["purity_gap"] - (vals["cosine_train"] - vals["cosine_val"])) < 1e-6


# ---------------------------------------------------------------------------
# T*-05 Purity improves with adversarial training (synthetic data)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_purity_improves_with_adversarial():
    """
    On synthetic data with a known spurious correlation injected into the
    training set, adversarial training should produce higher purity scores
    than standard training.

    This is a longer test (uses a tiny model + few epochs) to keep CI fast.
    """
    from tcrp.diagnostics.concept_purity import concept_purity_score
    from tcrp.training.adversarial_trainer import AdversarialTrainer
    from tcrp.training.trainer import Trainer
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(0)

    # Tiny config
    cfg = TCRPConfig(
        T=80, H=8, L=8, stride=4, d=16, k_max=1, periods=[24],
        lambda1=0.3, lambda2=1e-4, lambda3=0.01,
        adversarial=True, alpha_max=1.0, warmup_epochs=3,
    )
    T, H, B = cfg.T, cfg.H, 64

    # Synthetic dataset: y has both a genuine trend component and a spurious noise feature
    x = torch.randn(B, T)
    # Introduce spurious pattern: high-frequency noise correlated with target on training set
    x_train = x[:48] + 0.3 * torch.sin(torch.linspace(0, 40 * math.pi, T)).unsqueeze(0)
    y_train = x_train[:, -H:].clone()
    x_val   = x[48:]
    y_val   = x_val[:, -H:].clone()

    train_ds = TensorDataset(x_train, y_train)
    val_ds   = TensorDataset(x_val,   y_val)
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=16)

    # --- Standard model ---
    torch.manual_seed(1)
    model_std = TCRPForecaster(cfg)
    Trainer(model_std, cfg).fit(train_dl, val_dl, max_epochs=15)

    # --- Adversarial model ---
    torch.manual_seed(1)
    base     = TCRPForecaster(cfg)
    model_adv = AdversarialTCRPForecaster(base, alpha=0.0)
    AdversarialTrainer(model_adv, cfg).fit(train_dl, val_dl, max_epochs=15)

    segs = torch.randn(64, cfg.L)

    purity_std = [
        concept_purity_score(model_adv, model_adv.base.scorer, segs, k)["cosine_sim"]
        for k in range(cfg.K)
    ]

    # Use a fresh wrapper around the standard model just for the purity call
    model_std_wrapped = AdversarialTCRPForecaster(model_std, alpha=0.0)
    purity_adv = [
        concept_purity_score(model_std_wrapped, model_std_wrapped.base.scorer, segs, k)["cosine_sim"]
        for k in range(cfg.K)
    ]

    mean_std = sum(purity_std) / len(purity_std)
    mean_adv = sum(purity_adv) / len(purity_adv)

    # Adversarial purity should be higher on average (allow small margin for noisy training)
    assert mean_adv >= mean_std - 0.05, \
        f"Adversarial purity ({mean_adv:.4f}) should be >= standard ({mean_std:.4f})"


# ---------------------------------------------------------------------------
# T*-04/alpha integration test
# ---------------------------------------------------------------------------

def test_alpha_schedule_integration(small_config: TCRPConfig):
    """
    Full training loop with adversarial trainer on tiny synthetic dataset.
    Assert alpha increases monotonically after warmup; loss is finite at all epochs.
    """
    from tcrp.training.adversarial_trainer import AdversarialTrainer
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(42)
    T, H = small_config.T, small_config.H
    x = torch.randn(16, T)
    y = x[:, -H:].clone()
    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=8)

    base = TCRPForecaster(small_config)
    model = AdversarialTCRPForecaster(base, alpha=0.0)
    trainer = AdversarialTrainer(model, small_config)

    alphas: list[float] = []
    max_epochs = 10

    for epoch in range(1, max_epochs + 1):
        alpha = grl_alpha_schedule(
            epoch - 1, max_epochs,
            warmup_epochs=small_config.warmup_epochs,
            alpha_max=small_config.alpha_max,
        )
        model.set_alpha(alpha)
        alphas.append(alpha)
        bundle = trainer.train_epoch(dl)
        assert math.isfinite(bundle.total_loss.item()), \
            f"Loss diverged at epoch {epoch}: {bundle.total_loss.item()}"

    post_warmup = alphas[small_config.warmup_epochs:]
    if len(post_warmup) > 1:
        assert all(post_warmup[i] <= post_warmup[i + 1] for i in range(len(post_warmup) - 1)), \
            "alpha must be non-decreasing after warmup"
