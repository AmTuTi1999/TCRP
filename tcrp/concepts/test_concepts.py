"""
Phase 1 Concept Tests
=====================
T-03c · Volatility           — sigma_tilde, mu_v, psi
T-03d · Autocorrelation      — rho, theta_hat, z
T-03e · Structural Breaks    — b_mu, b_sigma, b_mu_tilde
T-03f · Distributional Shape — varsigma, kappa4, j
T-04  · Full Concept Vector  — ConceptScorer, K=22, gradcheck
"""

import math

import torch
from torch import Tensor
from torch.autograd import gradcheck

from tcrp.concepts.autocorrelation import autocorrelation_scores
from tcrp.concepts.concept_vector import ConceptScorer
from tcrp.concepts.distribution_shape import shape_scores
from tcrp.concepts.structural_breaks import break_scores
from tcrp.concepts.volatility import volatility_scores

# ===========================================================================
# Shared helpers
# ===========================================================================


def _series_from_deltas(deltas: Tensor) -> Tensor:
    return torch.cat([torch.zeros(1), deltas.cumsum(0)])


def _ar1(n: int, phi: float, seed: int = 0) -> Tensor:
    """AR(1) in the level space — used for OU / theta tests."""
    torch.manual_seed(seed)
    x = torch.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + torch.randn(1).item()
    return x


def _ar1_increments(n: int, phi: float, seed: int = 0) -> Tensor:
    """Series whose *increments* follow AR(1): delta[t] = phi*delta[t-1] + eps.
    The lag-1 ACF of delta equals phi, so rho_1 ≈ phi.
    """
    torch.manual_seed(seed)
    delta = torch.zeros(n - 1)
    delta[0] = torch.randn(1).item()
    for t in range(1, n - 1):
        delta[t] = phi * delta[t - 1] + torch.randn(1).item() * math.sqrt(1 - phi**2)
    s = torch.zeros(n)
    s[1:] = delta.cumsum(0)
    return s


def _ou_path(
    n: int,
    theta: float,
    mu: float = 0.0,
    sigma: float = 1.0,
    dt: float = 0.01,
    seed: int = 0,
) -> Tensor:
    torch.manual_seed(seed)
    x = torch.zeros(n)
    for t in range(1, n):
        dW = math.sqrt(dt) * torch.randn(1).item()
        x[t] = x[t - 1] + theta * (mu - x[t - 1]) * dt + sigma * dW
    return x


def _garch_path(
    n: int, omega: float = 0.1, alpha_g: float = 0.3, beta_g: float = 0.6, seed: int = 0
) -> Tensor:
    torch.manual_seed(seed)
    x = torch.zeros(n)
    h = omega / (1 - alpha_g - beta_g + 1e-6)
    for t in range(1, n):
        eps = math.sqrt(h) * torch.randn(1).item()
        x[t] = x[t - 1] + eps
        h = omega + alpha_g * eps**2 + beta_g * h
    return x


def _positively_clustered_d2(n: int, seed: int = 0) -> Tensor:
    torch.manual_seed(seed)
    phi = 0.7
    d2 = torch.zeros(n)
    d2[0] = torch.rand(1).item() + 0.1
    for t in range(1, n):
        d2[t] = phi * d2[t - 1] + (1 - phi) * torch.rand(1).item() + 0.05
    signs = torch.sign(torch.randn(n))
    delta = signs * d2.sqrt()
    s = torch.zeros(n + 1)
    s[1:] = delta.cumsum(0)
    return s


def _vol_break_series(
    n: int, low_std: float = 0.1, high_std: float = 2.0, seed: int = 0
) -> Tensor:
    torch.manual_seed(seed)
    mid = n // 2
    s1 = (torch.randn(mid) * low_std).cumsum(0)
    s2 = s1[-1] + (torch.randn(n - mid) * high_std).cumsum(0)
    return torch.cat([s1, s2])


def _step_series_clean(n: int, shift: float = 5.0) -> Tensor:
    s = torch.zeros(n)
    s[n // 2 :] = shift
    return s


def _reversal_series(n: int) -> Tensor:
    """Amplitude of 50 gives step ~1.0 so soft_monotonicity registers clear ±1."""
    mid = n // 2
    return torch.cat(
        [torch.linspace(0.0, 50.0, mid), torch.linspace(50.0, 0.0, n - mid)]
    )


def _linear_series(n: int, slope: float = 1.0) -> Tensor:
    return torch.linspace(0.0, slope * n, n)


def _right_clipped_series(n: int, seed: int = 0) -> Tensor:
    torch.manual_seed(seed)
    delta = torch.randn(n)
    delta = delta.clamp(max=delta.std().item() * 0.5)
    return _series_from_deltas(delta)


def _cauchy_series(n: int, seed: int = 0) -> Tensor:
    torch.manual_seed(seed)
    u = torch.rand(n) * (1 - 2e-4) + 1e-4
    delta = torch.tan(math.pi * (u - 0.5)).clamp(-50.0, 50.0)
    return _series_from_deltas(delta)


def _series_with_spike(n: int, spike_sigma: float = 5.0, seed: int = 0) -> Tensor:
    torch.manual_seed(seed)
    delta = torch.randn(n)
    delta[n // 2] = spike_sigma * delta.std().item()
    return _series_from_deltas(delta)


# ===========================================================================
# T-03c · Volatility
# ===========================================================================


def test_realvol_scale():
    """Doubling all increments doubles sigma_tilde."""
    torch.manual_seed(0)
    s = torch.randn(100).cumsum(0)
    delta = s[1:] - s[:-1]
    s2 = torch.zeros_like(s)
    s2[1:] = (2 * delta).cumsum(0)
    ratio = volatility_scores(s2, train_std=1.0)["sigma_tilde"].item() / (
        volatility_scores(s, train_std=1.0)["sigma_tilde"].item() + 1e-12
    )
    assert abs(ratio - 2.0) < 1e-4


def test_realvol_train_std_scaling():
    """Doubling train_std halves sigma_tilde."""
    torch.manual_seed(1)
    s = torch.randn(80).cumsum(0)
    ratio = volatility_scores(s, train_std=1.0)["sigma_tilde"].item() / (
        volatility_scores(s, train_std=2.0)["sigma_tilde"].item() + 1e-12
    )
    assert abs(ratio - 2.0) < 1e-4


def test_realvol_nonnegative():
    for seed in range(50):
        torch.manual_seed(seed)
        assert volatility_scores(torch.randn(60))["sigma_tilde"].item() >= 0.0


def test_realvol_batched_shape():
    assert volatility_scores(torch.randn(8, 64))["sigma_tilde"].shape == (8,)


def test_voltend_rising():
    """GARCH path → mu_v > 0 on average (volatility rising)."""
    mu_vs = [
        volatility_scores(
            _garch_path(200, omega=0.05, alpha_g=0.4, beta_g=0.55, seed=s)
        )["mu_v"].item()
        for s in range(100)
    ]
    assert sum(mu_vs) / len(mu_vs) > 0.0


def test_voltend_range():
    for seed in range(50):
        torch.manual_seed(seed)
        v = volatility_scores(torch.randn(80))["mu_v"].item()
        assert -1.0 < v < 1.0


def test_voltend_constant_series():
    assert not volatility_scores(torch.ones(50))["mu_v"].isnan().any()


def test_arch_clustering():
    """Positively clustered squared increments → psi > 0.1 on average."""
    psis = [
        volatility_scores(_positively_clustered_d2(200, seed=s))["psi"].item()
        for s in range(50)
    ]
    assert sum(psis) / len(psis) > 0.1


def test_arch_iid():
    """i.i.d. Gaussian increments → mean |psi| < 0.15."""
    psis = []
    for seed in range(500):
        torch.manual_seed(seed)
        psis.append(volatility_scores(torch.randn(100).cumsum(0))["psi"].item())
    assert sum(abs(p) for p in psis) / len(psis) < 0.15


def test_arch_range():
    for seed in range(50):
        torch.manual_seed(seed)
        v = volatility_scores(torch.randn(80))["psi"].item()
        assert -1.0 <= v <= 1.0


def test_arch_degenerate():
    assert (
        abs(volatility_scores(torch.arange(50, dtype=torch.float))["psi"].item()) < 1e-6
    )


def test_vol_differentiable_sigma():
    s = torch.randn(60, requires_grad=True)
    volatility_scores(s)["sigma_tilde"].backward()
    assert s.grad is not None


def test_vol_differentiable_mu_v():
    s = torch.randn(60, requires_grad=True)
    volatility_scores(s)["mu_v"].backward()
    assert s.grad is not None


def test_vol_differentiable_psi():
    s = torch.randn(60, requires_grad=True)
    volatility_scores(s)["psi"].backward()
    assert s.grad is not None


# ===========================================================================
# T-03d · Autocorrelation
# ===========================================================================


def test_rho_iid():
    """Random walk (i.i.d. increments) → mean |rho_k| < 0.15.

    A raw i.i.d. level series must NOT be used here: consecutive differences
    delta[t] = s[t]-s[t-1] share the term s[t], so their lag-1 ACF is -0.5.
    A random walk (cumsum of i.i.d. noise) has i.i.d. increments → ACF ≈ 0.
    """
    abs_rhos = []
    for seed in range(500):
        torch.manual_seed(seed)
        abs_rhos.append(
            autocorrelation_scores(torch.randn(100).cumsum(0), k_max=2)["rho"].abs()
        )
    assert torch.stack(abs_rhos).mean(dim=0).lt(0.15).all()


def test_rho_momentum():
    """AR(1) increments with phi=0.8 → rho_1 ≈ 0.8 (within ±0.1).

    AR(1) in the LEVEL space gives lag-1 increment ACF ≈ -(1-phi)/2 ≈ -0.1,
    not 0.8.  We need AR(1) in the INCREMENT space so that rho_1 = phi.
    """
    rho_1 = autocorrelation_scores(_ar1_increments(2000, phi=0.8, seed=42), k_max=2)[
        "rho"
    ][0].item()
    assert abs(rho_1 - 0.8) < 0.1


def test_rho_shape_unbatched():
    assert autocorrelation_scores(torch.randn(50), k_max=3)["rho"].shape == (3,)


def test_rho_shape_batched():
    assert autocorrelation_scores(torch.randn(4, 50), k_max=3)["rho"].shape == (4, 3)


def test_rho_range():
    assert (
        autocorrelation_scores(_ar1(200, phi=0.9), k_max=5)["rho"].abs() <= 1.0
    ).all()


def test_theta_ornstein_uhlenbeck():
    """OU path → theta_hat_cont within 20% of true theta."""
    true_theta = 2.0
    s = _ou_path(1000, theta=true_theta, dt=0.01, seed=7)
    theta_hat_cont = autocorrelation_scores(s, k_max=1)["theta_hat"].item() / 0.01
    assert abs(theta_hat_cont - true_theta) / true_theta < 0.20


def test_theta_random_walk():
    """Random walk → mean theta_hat ≈ 0."""
    theta_hats = []
    for seed in range(200):
        torch.manual_seed(seed)
        s = torch.cumsum(torch.randn(200), dim=0)
        theta_hats.append(autocorrelation_scores(s, k_max=1)["theta_hat"].item())
    assert abs(sum(theta_hats) / len(theta_hats)) < 0.3


def test_theta_clamp():
    for seed in range(50):
        torch.manual_seed(seed)
        v = autocorrelation_scores(torch.randn(80), k_max=1)["theta_hat"].item()
        assert -2.0 <= v <= 2.0


def test_z_range():
    for seed in range(200):
        torch.manual_seed(seed)
        v = autocorrelation_scores(torch.randn(100) * 10 + 5)["z"].item()
        assert -3.0 <= v <= 3.0


def test_z_batched_range():
    torch.manual_seed(0)
    z = autocorrelation_scores(torch.randn(16, 64) * 5)["z"]
    assert (z >= -3.0).all() and (z <= 3.0).all()


def test_z_constant_series():
    assert abs(autocorrelation_scores(torch.ones(50))["z"].item()) < 1e-4


def test_acf_exclude_rho():
    out = autocorrelation_scores(torch.randn(50), include_rho=False)
    assert "rho" not in out and "theta_hat" in out and "z" in out


def test_acf_exclude_theta():
    out = autocorrelation_scores(torch.randn(50), include_theta=False)
    assert "theta_hat" not in out and "rho" in out and "z" in out


def test_acf_exclude_z():
    out = autocorrelation_scores(torch.randn(50), include_z=False)
    assert "z" not in out and "rho" in out and "theta_hat" in out


def test_acf_all_excluded():
    assert (
        autocorrelation_scores(
            torch.randn(50), include_rho=False, include_theta=False, include_z=False
        )
        == {}
    )


def test_acf_default_all_included():
    assert set(autocorrelation_scores(torch.randn(50)).keys()) == {
        "rho",
        "theta_hat",
        "z",
    }


def test_acf_differentiable_rho():
    s = torch.randn(60, requires_grad=True)
    autocorrelation_scores(s, k_max=2)["rho"].sum().backward()
    assert s.grad is not None


def test_acf_differentiable_theta():
    s = torch.randn(60, requires_grad=True)
    autocorrelation_scores(s, k_max=1)["theta_hat"].backward()
    assert s.grad is not None


def test_acf_differentiable_z():
    s = torch.randn(60, requires_grad=True)
    autocorrelation_scores(s, k_max=1)["z"].backward()
    assert s.grad is not None


# ===========================================================================
# T-03e · Structural Breaks
# ===========================================================================


def test_level_shift():
    """Clean step function → b_mu > 0.8."""
    assert break_scores(_step_series_clean(100, shift=5.0))["b_mu"].item() > 0.8


def test_level_shift_robust():
    """Noisy step → mean b_mu > 0.8."""
    vals = []
    for seed in range(20):
        torch.manual_seed(seed)
        s = torch.zeros(100)
        s[50:] = 5.0
        s = s + torch.randn(100) * 0.2
        vals.append(break_scores(s)["b_mu"].item())
    assert sum(vals) / len(vals) > 0.8


def test_no_break_level():
    torch.manual_seed(0)
    assert break_scores(torch.randn(100))["b_mu"].item() < 0.3


def test_level_range():
    for seed in range(50):
        torch.manual_seed(seed)
        assert 0.0 <= break_scores(torch.randn(80))["b_mu"].item() < 1.0


def test_level_batched_shape():
    assert break_scores(torch.randn(8, 64))["b_mu"].shape == (8,)


def test_vol_break():
    """Low-vol / high-vol halves → mean b_sigma > 0.5."""
    vals = [
        break_scores(_vol_break_series(100, seed=s))["b_sigma"].item()
        for s in range(20)
    ]
    assert sum(vals) / len(vals) > 0.5


def test_no_break_flat():
    """Constant segment → all scores ≈ 0."""
    out = break_scores(torch.ones(100))
    assert abs(out["b_mu"].item()) < 1e-4
    assert abs(out["b_sigma"].item()) < 1e-4
    assert abs(out["b_mu_tilde"].item()) < 1e-4


def test_vol_degenerate_sigma1():
    s = torch.cat([torch.ones(50), torch.randn(50)])
    assert break_scores(s)["b_sigma"].item() == 0.0


def test_vol_range():
    for seed in range(50):
        torch.manual_seed(seed)
        assert 0.0 <= break_scores(torch.randn(80))["b_sigma"].item() < 1.0


def test_vol_batched_shape():
    assert break_scores(torch.randn(8, 64))["b_sigma"].shape == (8,)


def test_slope_break_reversal():
    """Rising then falling → b_mu_tilde < -0.5."""
    assert break_scores(_reversal_series(100))["b_mu_tilde"].item() < -0.5


def test_slope_break_continuation():
    """Uniformly rising → |b_mu_tilde| < 0.1."""
    assert abs(break_scores(_linear_series(100, slope=1.0))["b_mu_tilde"].item()) < 0.1


def test_slope_break_falling_continuation():
    assert abs(break_scores(_linear_series(100, slope=-1.0))["b_mu_tilde"].item()) < 0.1


def test_slope_break_acceleration():
    s = torch.cat([torch.zeros(50), torch.linspace(0.0, 5.0, 50)])
    assert break_scores(s)["b_mu_tilde"].item() > 0.0


def test_slope_range():
    for seed in range(50):
        torch.manual_seed(seed)
        v = break_scores(torch.randn(80))["b_mu_tilde"].item()
        assert -1.0 < v < 1.0


def test_slope_batched_shape():
    assert break_scores(torch.randn(8, 64))["b_mu_tilde"].shape == (8,)


def test_breaks_differentiable_b_mu():
    s = torch.randn(60, requires_grad=True)
    break_scores(s)["b_mu"].backward()
    assert s.grad is not None


def test_breaks_differentiable_b_sigma():
    s = torch.randn(60, requires_grad=True)
    break_scores(s)["b_sigma"].backward()
    assert s.grad is not None


def test_breaks_differentiable_b_mu_tilde():
    s = torch.randn(60, requires_grad=True)
    break_scores(s)["b_mu_tilde"].backward()
    assert s.grad is not None


def test_breaks_gradcheck():
    torch.manual_seed(0)
    s = torch.randn(20, dtype=torch.float64, requires_grad=True)

    def fn(x):
        out = break_scores(x)
        return torch.stack([out["b_mu"], out["b_sigma"], out["b_mu_tilde"]])

    assert gradcheck(fn, (s,), eps=1e-6, atol=1e-4, rtol=1e-3)


# ===========================================================================
# T-03f · Distributional Shape
# ===========================================================================


def test_skew_negative():
    """Right-clipped distribution → mean varsigma < 0."""
    vals = [
        shape_scores(_right_clipped_series(300, seed=s))["varsigma"].item()
        for s in range(100)
    ]
    assert sum(vals) / len(vals) < 0.0


def test_skew_gaussian():
    """Gaussian increments → mean |varsigma| < 0.3."""
    abs_skews = []
    for seed in range(500):
        torch.manual_seed(seed)
        abs_skews.append(
            abs(shape_scores(torch.randn(100).cumsum(0))["varsigma"].item())
        )
    assert sum(abs_skews) / len(abs_skews) < 0.3


def test_skew_clamp():
    for seed in range(50):
        torch.manual_seed(seed)
        v = shape_scores(torch.randn(80).cumsum(0))["varsigma"].item()
        assert -3.0 <= v <= 3.0


def test_skew_batched_shape():
    assert shape_scores(torch.randn(8, 64).cumsum(dim=-1))["varsigma"].shape == (8,)


def test_kurtosis_gaussian():
    """Gaussian increments → mean |kappa4| < 0.5."""
    abs_kurt = []
    for seed in range(500):
        torch.manual_seed(seed)
        abs_kurt.append(abs(shape_scores(torch.randn(100).cumsum(0))["kappa4"].item()))
    assert sum(abs_kurt) / len(abs_kurt) < 0.5


def test_kurtosis_cauchy():
    """Cauchy increments → mean kappa4 > 5 (near upper clamp)."""
    kurts = [
        shape_scores(_cauchy_series(200, seed=s))["kappa4"].item() for s in range(50)
    ]
    assert sum(kurts) / len(kurts) > 5.0


def test_kurtosis_clamp():
    for seed in range(50):
        torch.manual_seed(seed)
        k = shape_scores(torch.randn(80).cumsum(0))["kappa4"].item()
        assert -3.0 <= k <= 10.0


def test_kurtosis_batched_shape():
    assert shape_scores(torch.randn(8, 64))["kappa4"].shape == (8,)


def test_jump_detected():
    """5σ spike → j > 0.8."""
    for seed in range(20):
        j = shape_scores(
            _series_with_spike(100, spike_sigma=5.0, seed=seed),
            gamma_j=2.0,
            jump_threshold=3.0,
        )["j"].item()
        assert j > 0.8, f"seed={seed}: j={j:.4f}"


def test_jump_not_detected():
    """i.i.d. Gaussian series (29 increments ~ N(0,√2); 3σ threshold ≈ 4.2) → mean j < 0.3."""
    js = []
    for seed in range(500):
        torch.manual_seed(seed)
        js.append(
            shape_scores(torch.randn(30), gamma_j=2.0, jump_threshold=3.0)["j"].item()
        )
    assert sum(js) / len(js) < 0.3


def test_jump_range():
    for seed in range(50):
        torch.manual_seed(seed)
        j = shape_scores(torch.randn(80))["j"].item()
        assert 0.0 < j < 1.0


def test_jump_batched_shape():
    assert shape_scores(torch.randn(8, 64))["j"].shape == (8,)


def test_shape_gradcheck():
    """gradcheck in float64 on all three shape outputs."""
    torch.manual_seed(0)
    s = torch.randn(20, dtype=torch.float64, requires_grad=True)

    def fn(x):
        out = shape_scores(x, gamma_j=2.0, jump_threshold=3.0)
        return torch.stack([out["varsigma"], out["kappa4"], out["j"]])

    assert gradcheck(fn, (s,), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_shape_differentiable_varsigma():
    s = torch.randn(60, requires_grad=True)
    shape_scores(s)["varsigma"].backward()
    assert s.grad is not None


def test_shape_differentiable_kappa4():
    s = torch.randn(60, requires_grad=True)
    shape_scores(s)["kappa4"].backward()
    assert s.grad is not None


def test_shape_differentiable_j():
    s = torch.randn(60, requires_grad=True)
    shape_scores(s)["j"].backward()
    assert s.grad is not None


# ===========================================================================
# T-04 · Full Temporal Concept Vector
# ===========================================================================

_DEFAULT_K = 22  # 16 + k_max=2 + len(periods=[2,4,8,16])=4


def test_concept_scorer_output_shape():
    scorer = ConceptScorer()
    assert scorer.num_concepts == _DEFAULT_K
    s = torch.randn(4, 64)
    c = scorer(s)
    assert c.shape == (4, _DEFAULT_K)


def test_concept_scorer_names_length():
    scorer = ConceptScorer()
    names = scorer.concept_names
    assert len(names) == _DEFAULT_K


def test_concept_scorer_names_order():
    """Spot-check canonical column names at fixed indices."""
    scorer = ConceptScorer()
    names = scorer.concept_names
    assert names[0] == "mu_signed"
    assert names[1] == "mu_mag"
    assert names[2] == "kappa_signed"
    assert names[3] == "tau"
    assert names[4] == "xi"
    assert names[5] == "sigma_tilde"
    assert names[6] == "mu_v"
    assert names[7] == "psi"
    assert names[8] == "rho_1"
    assert names[9] == "rho_2"
    assert names[10] == "theta_hat"
    assert names[11] == "z"
    assert names[12] == "b_mu"
    assert names[13] == "b_sigma"
    assert names[14] == "b_mu_tilde"
    assert names[15] == "varsigma"
    assert names[16] == "kappa4"
    assert names[17] == "j"
    assert names[18] == "rho_p2"
    assert names[21] == "rho_p16"


def test_concept_scorer_custom_k_max():
    scorer = ConceptScorer(k_max=3, periods=[2, 4])
    assert scorer.num_concepts == 16 + 3 + 2
    assert len(scorer.concept_names) == scorer.num_concepts
    c = scorer(torch.randn(2, 64))
    assert c.shape == (2, scorer.num_concepts)


def test_concept_scorer_no_learned_params():
    scorer = ConceptScorer()
    assert sum(p.numel() for p in scorer.parameters()) == 0


def test_concept_scorer_requires_2d():
    scorer = ConceptScorer()
    try:
        scorer(torch.randn(64))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_concept_scorer_all_finite():
    """All concept values should be finite for typical random input."""
    scorer = ConceptScorer()
    torch.manual_seed(0)
    s = torch.randn(8, 64)
    c = scorer(s)
    assert torch.isfinite(c).all(), "Non-finite values in concept vector"


def test_concept_scorer_differentiable():
    scorer = ConceptScorer()
    s = torch.randn(2, 64, requires_grad=True)
    scorer(s).sum().backward()
    assert s.grad is not None


def test_concept_scorer_gradcheck():
    """gradcheck in float64 on the full K=22 concept vector.

    L=40 ensures period=16 has at least 2 full cycles in the FFT bins.
    """
    scorer = ConceptScorer()
    torch.manual_seed(0)
    s = torch.randn(1, 40, dtype=torch.float64, requires_grad=True)
    assert gradcheck(scorer, (s,), eps=1e-5, atol=1e-3, rtol=1e-2, raise_exception=True)
