# TCRP Implementation Tasks

Temporal Concept Relevance Propagation — forecasting architecture.  
Paper reference: `tcrp_v3.tex`.  
Stack: Python 3.11, PyTorch 2.x, numpy, scipy.

---

## Module layout

```
tcrp/
├── concepts/
│   ├── __init__.py
│   ├── monotonicity.py
│   ├── curvature.py
│   ├── periodicity.py
│   ├── stochasticity.py
│   ├── volatility.py
│   ├── autocorrelation.py
│   ├── breaks.py
│   ├── shape.py
│   └── concept_vector.py
├── model/
│   ├── __init__.py
│   ├── segmentation.py
│   ├── encoder.py
│   ├── bottleneck.py
│   ├── aggregation.py
│   ├── decoder.py
│   └── tcrp.py
├── training/
│   ├── __init__.py
│   ├── losses.py
│   └── trainer.py
├── analysis/
│   ├── __init__.py
│   ├── lrp.py
│   └── tcrp_analysis.py
├── data/
│   ├── __init__.py
│   ├── datasets.py
│   └── preprocessing.py
└── eval/
    ├── __init__.py
    ├── metrics.py
    └── baselines.py
```

---

## Phase 1 — Analytic Concept Scores
> `tcrp/concepts/`  
> These are pure functions: no learned parameters, no PyTorch modules.  
> All inputs are raw numpy arrays or torch tensors of shape `(L,)`.  
> All outputs must be differentiable w.r.t. the input values.

### T-01 · Soft monotonicity score
**File:** `concepts/monotonicity.py`  
**Paper:** Def. 1, Eq. 1–2

Implement `soft_monotonicity(s: Tensor, alpha: float = 5.0) -> Tensor`:
- Compute first differences `delta[l] = s[l] - s[l-1]` for `l = 1…L-1`
- Return `mu = sigmoid(alpha * delta).mean()` — shape `()`
- Return `mu_signed = 2 * mu - 1` — range `(-1, 1)`
- Return `mu_mag = mu_signed.abs()` — range `(0, 1)`
- All three must be returned together as a dict or named tuple
- Verify: constant-increasing input → `mu_signed ≈ 1`; constant-decreasing → `mu_signed ≈ -1`; white noise → `mu_signed ≈ 0`
- Verify: gradient w.r.t. `s` is everywhere finite and non-zero (use `torch.autograd.gradcheck`)

### T-02 · Soft curvature score and observed tendency
**File:** `concepts/curvature.py`  
**Paper:** Def. 2, Eq. 3–4

Implement `soft_curvature(s: Tensor, beta: float = 5.0) -> Tensor`:
- Compute second differences `delta2[l] = delta[l+1] - delta[l]` for `l = 1…L-2`
- Return `kappa_signed = 2 * sigmoid(beta * delta2).mean() - 1` — range `(-1, 1)`
- Return `tau = mu_signed * kappa_signed` — observed tendency, range `(-1, 1)`
- This function must accept the already-computed `mu_signed` from T-01 to avoid recomputing first differences
- Verify four-regime table (Table 1 in paper): accelerating rise → `tau > 0`; decelerating rise → `tau < 0`; accelerating fall → `tau < 0`; decelerating fall → `tau > 0`

### T-03 · Periodicity score
**File:** `concepts/periodicity.py`  
**Paper:** Def. 3, Eq. 5

Implement `periodicity_score(s: Tensor, periods: list[int]) -> Tensor`:
- Compute DFT via `torch.fft.rfft(s)`; power spectrum `P[nu] = |X[nu]|^2`
- Total non-DC power = `P[1:].sum()`; DC component is `P[0]` — excluded from denominator
- For each `p` in `periods`, find bin `nu_p = round(L / p)`, return `rho_p = P[nu_p] / total_power`
- Return tensor of shape `(len(periods),)` with all `rho_p` values
- Clamp to `[0, 1]` after division to handle numerical edge cases
- Verify: pure sinusoid at period `p` → `rho_p ≈ 1.0`; white noise → all `rho_p ≈ 1 / (L//2)`
- Note: `round(L / p)` must be clamped to `[1, L//2]` to stay in valid DFT range

### T-03b · Stochasticity score (Brownian motion similarity)
**File:** `concepts/stochasticity.py`  
**Paper:** Def. 4, Eqs. 10–13

Implement `stochasticity_score(s: Tensor, gamma: float = 0.5, weights: tuple = (1/3, 1/3, 1/3)) -> Tensor` returning scalar `xi` in `[0, 1]`.

The score is a weighted geometric mean of three sub-scores, each testing one property of a standard Brownian motion (Wiener process).

**Sub-score (a): Hurst similarity `phi_H`**

A standard Brownian motion has Hurst exponent H = 0.5.  Estimate H via variance scaling:
- Compute first differences `delta = s[1:] - s[:-1]`, shape `(L-1,)`
- Split into first half `delta[:L//2]` and full sequence `delta`
- `H_hat = 0.5 + (log(var(delta[:L//2])) - log(var(delta))) / (2 * log(2))`
- Clamp `H_hat` to `(0, 1)` before computing similarity
- `phi_H = 1 - 2 * |H_hat - 0.5|` — range `[0, 1]`
- Differentiable through `torch.var` (use `unbiased=False` for numerical stability)
- Degenerate case: if `var(delta) < 1e-8` (constant segment), return `phi_H = 0.0`

**Sub-score (b): Spectral flatness `phi_flat`**

Brownian motion increments are white noise → flat power spectrum:
- Compute power spectrum of increments `P = |fft(delta)|^2`, shape `(L//2,)` (positive freqs only)
- Clamp `P` to minimum `1e-10` before log to avoid `-inf`
- Geometric mean of spectrum: `G = exp(mean(log(P)))`
- Arithmetic mean: `A = mean(P)`
- `phi_flat = G / A` — this is the Wiener entropy ratio, range `(0, 1]`
- Verify: white noise input → `phi_flat ≈ 1`; pure sinusoid input → `phi_flat ≈ 0`

**Sub-score (c): Increment Gaussianity `phi_kurt`**

Brownian increments are i.i.d. Gaussian → excess kurtosis = 0:
- Compute excess kurtosis of `delta`: `kurt4 = mean((delta - mean(delta))^4) / var(delta)^2 - 3`
- `phi_kurt = exp(-gamma * |kurt4|)` — range `(0, 1]`
- Degenerate case: if `var(delta) < 1e-8`, return `phi_kurt = 0.0`
- Gaussian white noise → `kurt4 ≈ 0` → `phi_kurt ≈ 1`
- Heavy-tailed noise or impulses → `|kurt4| >> 0` → `phi_kurt → 0`

**Composite score:**
- `xi = phi_H^w1 * phi_flat^w2 * phi_kurt^w3`
- Compute in log space for numerical stability: `log_xi = w1*log(phi_H + eps) + w2*log(phi_flat + eps) + w3*log(phi_kurt + eps)`, then `xi = exp(log_xi)`
- Clamp final `xi` to `[0, 1]`

**Tests:**
- Brownian motion simulation (use `torch.cumsum(torch.randn(L), dim=0)`) → `xi` should be consistently above 0.5 (not necessarily close to 1 due to finite-sample variance)
- Perfectly linear segment → `phi_H → 1` (H ≈ 1) → `phi_H ≈ 0` → `xi ≈ 0`
- Pure sinusoid → `phi_flat ≈ 0` → `xi ≈ 0`
- Cauchy-noise increments → `|kurt4| >> 0` → `phi_kurt ≈ 0` → `xi ≈ 0`
- Run `gradcheck` in double precision — all three sub-scores must have finite, non-zero gradients
- Verify geometric mean property: if any one sub-score is near 0, `xi` must also be near 0 regardless of others

### T-03c · Volatility concepts
**File:** `concepts/volatility.py`  
**Paper:** Def. 5, Eqs. 14–16

Implement `volatility_scores(s: Tensor, train_std: float = 1.0) -> dict`:

**Realised volatility `sigma_tilde`:**
- `sigma = sqrt(mean(delta^2))` where `delta = s[1:] - s[:-1]`
- Normalise: `sigma_tilde = sigma / train_std` — dimensionless, range `[0, ∞)`
- `train_std` passed in from dataset statistics computed on training set only

**Volatility trend `mu_v`:**
- Compute `v = |delta|` — absolute increment sequence, shape `(L-1,)`
- Apply `soft_monotonicity(v, alpha)` from T-01 → returns `mu_v_signed ∈ (-1, 1)`
- `mu_v > 0`: volatility rising (entering high-vol regime); `mu_v < 0`: falling

**ARCH clustering `psi`:**
- `psi = pearson_corr(delta[:-1]^2, delta[1:]^2)` — lag-1 autocorrelation of squared increments
- Implement as: `d2 = delta^2; psi = corr(d2[:-1], d2[1:])` via `torch`-differentiable Pearson
- Range `(-1, 1)`; degenerate case `var(d2) < 1e-8` → return `psi = 0.0`

Returns dict `{sigma_tilde, mu_v, psi}`.

Tests:
- `test_realvol_scale`: doubling all increments doubles `sigma`
- `test_voltend_rising`: GARCH-simulated increasing-variance path → `mu_v > 0` on average
- `test_arch_clustering`: squared increments with positive ACF → `psi > 0`
- `test_arch_iid`: i.i.d. Gaussian increments → `psi ≈ 0` on average over 500 trials

---

### T-03d · Autocorrelation concepts
**File:** `concepts/autocorrelation.py`  
**Paper:** Def. 6, Eqs. 17–19

Implement `autocorrelation_scores(s: Tensor, k_max: int = 2) -> dict`:

**Lag-k increment ACF `rho_k`:**
- `delta = s[1:] - s[:-1]`, shape `(L-1,)`
- For each `k in 1..k_max`: `rho_k = pearson_corr(delta[:-k], delta[k:])` — range `(-1, 1)`
- Return as tensor of shape `(k_max,)` — differentiable through Pearson

**Mean-reversion speed `theta_hat`:**
- `theta_hat = -corr(s[:-1], delta) * std(delta) / (std(s[:-1]) + eps)`
- This is the OLS coefficient from regressing `delta` on `s[:-1]` (Ornstein-Uhlenbeck estimator)
- Range approximately `[0, ∞)` for mean-reverting; clamp output to `[-2, 2]` for stability
- `theta_hat > 0`: mean-reverting; `theta_hat ≤ 0`: trending or random walk

**Z-score `z`:**
- `z = (s[-1] - s.mean()) / (s.std() + eps)` — last observation relative to within-segment mean
- Clamp to `[-3, 3]`

Returns dict `{rho: Tensor(k_max), theta_hat, z}`.

Tests:
- `test_rho_iid`: i.i.d. series → all `|rho_k| < 0.15` on average over 500 trials
- `test_rho_momentum`: AR(1) with coefficient 0.8 → `rho_1 ≈ 0.8`
- `test_theta_ornstein_uhlenbeck`: OU path with known `theta` → estimated `theta_hat` within 20% of true value over 1000-point path
- `test_theta_random_walk`: random walk → `theta_hat ≈ 0`
- `test_z_range`: all outputs in `[-3, 3]`

---

### T-03e · Structural break concepts
**File:** `concepts/breaks.py`  
**Paper:** Def. 7, Eqs. 20–22

Implement `break_scores(s: Tensor) -> dict`:

Split segment into `s1 = s[:L//2]`, `s2 = s[L//2:]`.

**Mean break `b_mu`:**
- `sigma_hat = std(delta)` where `delta = s[1:] - s[:-1]` — full-window increment std
- `b_mu = tanh(|mean(s2) - mean(s1)| / (sigma_hat + eps))` — range `[0, 1)`
- Degenerate: if `sigma_hat < 1e-8` → `b_mu = 0.0`

**Variance break `b_sigma`:**
- `sigma1 = std(s1[1:] - s1[:-1])`, `sigma2 = std(s2[1:] - s2[:-1])`
- `b_sigma = tanh(|log(sigma2 / (sigma1 + eps))|)` — range `[0, 1)`
- Degenerate: if `sigma1 < 1e-8 or sigma2 < 1e-8` → `b_sigma = 0.0`

**Slope break `b_mu_tilde`:**
- Apply `soft_monotonicity` to `s1` → `mu1_signed`; to `s2` → `mu2_signed`
- `b_mu_tilde = (mu2_signed - mu1_signed) / 2.0` — range `(-1, 1)`

Returns dict `{b_mu, b_sigma, b_mu_tilde}`.

Tests:
- `test_level_shift`: step function (constant then shifted by 5σ) → `b_mu > 0.8`
- `test_no_break_flat`: constant segment → all break scores ≈ 0
- `test_vol_break`: low-vol first half, high-vol second half → `b_sigma > 0.5`
- `test_slope_break_reversal`: rising then falling → `b_mu_tilde < -0.5`
- `test_slope_break_continuation`: uniformly rising → `|b_mu_tilde| < 0.1`

---

### T-03f · Distributional shape concepts
**File:** `concepts/shape.py`  
**Paper:** Def. 8, Eqs. 23–25

Implement `shape_scores(s: Tensor, gamma_j: float = 2.0, jump_threshold: float = 3.0) -> dict`:

`delta = s[1:] - s[:-1]`; `delta_c = delta - delta.mean()` (centred)

**Signed skewness `varsigma`:**
- `varsigma = mean(delta_c^3) / (std(delta)^3 + eps)`
- Clamp to `[-3, 3]`
- Negative: crash risk; positive: right tail / lottery profile

**Signed excess kurtosis `kappa4`:**
- `kappa4 = mean(delta_c^4) / (std(delta)^4 + eps) - 3`
- Clamp to `[-3, 10]` (excess kurtosis rarely negative beyond -3)
- Positive: heavy tails; negative: thin tails

**Soft jump indicator `j`:**
- `j = sigmoid(gamma_j * (max(|delta|) - jump_threshold * std(delta)))`
- Range `(0, 1)`; `j > 0.5` when max increment exceeds `3σ`
- Differentiable through `sigmoid` and `max` (use `delta.abs().max()`)

Returns dict `{varsigma, kappa4, j}`.

Tests:
- `test_skew_negative`: right-clipped distribution → `varsigma < 0`
- `test_skew_gaussian`: Gaussian increments → `|varsigma| < 0.3` on average
- `test_kurtosis_gaussian`: Gaussian → `|kappa4| < 0.5` on average over 500 trials
- `test_kurtosis_cauchy`: Cauchy increments → `kappa4 >> 0` (clamped to 10)
- `test_jump_detected`: segment with one 5σ increment → `j > 0.8`
- `test_jump_not_detected`: Gaussian segment → `j < 0.3` on average
- `test_gradcheck`: `gradcheck` in double precision on all three outputs

---

### T-04 · Full temporal concept vector  *(updated)*
**File:** `concepts/concept_vector.py`  
**Paper:** Eq. 26, Table 2

Implement `ConceptScorer(alpha, beta, periods, gamma=0.5, gamma_j=2.0, k_max=2, jump_threshold=3.0, train_std=1.0)` as `nn.Module` with no learned parameters:

- `forward(s: Tensor) -> Tensor` where `s` has shape `(B, L)` for batched segments
- Returns concept vector `c` of shape `(B, K)` where `K = 16 + k_max + len(periods)`
- **Column order (canonical):**

```
[mu_signed, mu_mag,                    # trend (0:2)
 kappa_signed, tau,                    # curvature (2:4)
 xi,                                   # stochasticity (4)
 sigma_tilde, mu_v, psi,              # volatility (5:8)
 rho_1, ..., rho_k_max,               # increment ACF (8:8+k_max)
 theta_hat, z,                         # mean reversion (8+k_max:10+k_max)
 b_mu, b_sigma, b_mu_tilde,           # breaks (10+k_max:13+k_max)
 varsigma, kappa4, j,                  # shape (13+k_max:16+k_max)
 rho_p1, ..., rho_pM]                 # periodicity (16+k_max:)
```

- `concept_names` property returns the above as a list of strings
- All columns differentiable w.r.t. `s`
- `gradcheck` on full vector in double precision with default K=22


---

## Phase 2 — Segmentation
> `tcrp/model/segmentation.py`

### T-05 · Sliding window segmentation
**Paper:** Eq. 7 (background notation)

Implement `Segmenter(L: int, stride: int)` as an `nn.Module`:
- `forward(x: Tensor) -> Tensor` where `x` has shape `(B, T)` (univariate)
- Returns segments of shape `(B, N, L)` where `N = floor((T - L) / stride) + 1`
- Use `x.unfold(dimension=1, size=L, step=stride)` — no data copy
- Store `t_n = [n * stride for n in range(N)]` as `self.start_indices`
- Store `overlap_counts[t]` as a `(T,)` tensor: how many segments contain timestep `t`
- Raise `ValueError` if `T < L`
- Verify that every timestep is covered by at least one segment when `stride <= L`

---

## Phase 3 — Shared Temporal Encoder
> `tcrp/model/encoder.py`

### T-06 · Dilated causal TCN block
**Paper:** Appendix B (encoder description)

Implement `CausalDilatedBlock(in_ch, out_ch, kernel_size, dilation)` as `nn.Module`:
- Causal padding: `pad = (kernel_size - 1) * dilation`; apply left-only via `F.pad(x, (pad, 0))`
- Conv1d → WeightNorm → ReLU → Conv1d → WeightNorm → ReLU
- Residual connection with `1×1` projection if `in_ch != out_ch`
- No future timesteps may be seen: verify causality by checking that `output[:, :, t]` does not depend on `input[:, :, t+1:]`

### T-07 · Stacked TCN encoder
**Paper:** Appendix B

Implement `TCNEncoder(in_ch=1, hidden=64, n_layers=4, kernel_size=3)` as `nn.Module`:
- Stack four `CausalDilatedBlock` with dilations `[1, 2, 4, 8]`
- Final mean-pool over the time dimension: `z = out.mean(dim=-1)` → shape `(B*N, d)`
- `forward(segments: Tensor) -> Tensor`:
  - Input shape `(B, N, L)` — reshape to `(B*N, 1, L)` before passing through TCN
  - Output shape `(B, N, d)`
- Receptive field of stacked dilations `[1,2,4,8]` with kernel 3 = 30 timesteps; assert `L >= 30` or warn
- Weight sharing is automatic because we flatten `(B, N)` into the batch dimension

---

## Phase 4 — Concept Projection Bottleneck
> `tcrp/model/bottleneck.py`

### T-08 · Linear concept projection
**Paper:** Eq. 8

Implement `ConceptProjection(d: int, K: int)` as `nn.Module`:
- Single `nn.Linear(d, K, bias=True)` — weight matrix rows are the concept directions `w_k`
- `forward(Z: Tensor) -> Tensor`:
  - Input `Z` shape `(B, N, d)`
  - Output `A` shape `(B, N, K)` — concept activation matrix
- Initialise weight rows to top-K PCA directions of a calibration batch (implement `init_from_pca(Z_calib: Tensor)` method)
- After forward, `A` replaces `Z` for all downstream computation — enforce this by not returning `Z`

### T-09 · Concept alignment loss
**Paper:** Eq. 9

Implement `alignment_loss(A: Tensor, C: Tensor) -> Tensor`:
- `A` shape `(B, N, K)` — learned concept activations
- `C` shape `(B, N, K)` — analytic concept scores from `ConceptScorer`
- For each concept `k`, compute Pearson correlation between `A[:,:,k].flatten()` and `C[:,:,k].flatten()` across the batch
- Return `sum over k of (1 - corr_k)^2` — scalar loss
- Handle the degenerate case where `C[:,:,k]` has zero variance (constant concept score across all segments): skip that concept's term or clamp correlation to 0
- Must be differentiable w.r.t. `A` (and through `A` back to encoder weights)
- Unit test: when `A == C`, loss should be exactly 0.0

---

## Phase 5 — Temporal Concept Aggregation
> `tcrp/model/aggregation.py`

### T-10 · Additive attention pooling
**Paper:** Eq. 10

Implement `ConceptAttentionPool(K: int, hidden: int = 32)` as `nn.Module`:
- Attention energy: `e_n = v^T tanh(U @ A_n)` where `U: (hidden, K)`, `v: (hidden,)`
- Attention weights: `eta = softmax(e)` over the `N` dimension — shape `(B, N)`
- Pooled vector: `h = sum_n eta_n * A_n` — shape `(B, K)`
- `forward(A: Tensor) -> tuple[Tensor, Tensor]`:
  - Returns `(h, eta)` — both needed downstream; `eta` is used in TCRP analysis
- Store `eta` as `self.last_eta` after each forward pass for the analysis pass
- Temperature parameter `temp: float = 1.0` scaling the energies before softmax

---

## Phase 6 — Horizon Decoder
> `tcrp/model/decoder.py`

### T-11 · Linear horizon decoder
**Paper:** Eq. 11

Implement `HorizonDecoder(K: int, H: int)` as `nn.Module`:
- Single `nn.Linear(K, H, bias=True)` — matrix `Theta`, shape `(H, K)`
- `forward(h: Tensor) -> Tensor`:
  - Input `h` shape `(B, K)`
  - Output `y_hat` shape `(B, H)`
- Store decoder weights as `self.Theta` for direct access during TCRP analysis
- No non-linearity — linearity is the paper's deliberate design constraint

### T-12 · Probabilistic decoder (Gaussian)
**Paper:** §4.7

Implement `GaussianDecoder(K: int, H: int)` as `nn.Module`:
- Two linear heads: `mu_head: Linear(K, H)` and `log_sigma_head: Linear(K, H)`
- `forward(h) -> tuple[Tensor, Tensor]`: returns `(mu, sigma)` where `sigma = exp(log_sigma).clamp(min=1e-6)`
- `nll_loss(mu, sigma, y_true) -> Tensor`: negative Gaussian log-likelihood
- TCRP analysis propagates through `mu_head` only; `log_sigma_head` is analysed separately

---

## Phase 7 — Full TCRP Model
> `tcrp/model/tcrp.py`

### T-13 · TCRP forecaster (assembly)
**Paper:** §4

Implement `TCRPForecaster(config: TCRPConfig)` as `nn.Module`:

```python
@dataclass
class TCRPConfig:
    T: int              # look-back length
    H: int              # forecast horizon
    L: int = 20         # segment window length
    stride: int = 5     # segmentation stride
    d: int = 64         # encoder hidden dim
    K: int = 22         # 16 + k_max + M; see concept_vector.py canonical ordering
    k_max: int = 2      # number of ACF lags
    periods: list = field(default_factory=lambda: [24, 168])
    alpha: float = 5.0  # monotonicity temperature
    beta: float = 5.0   # curvature temperature
    gamma: float = 0.5  # stochasticity kurtosis decay
    gamma_j: float = 2.0  # jump indicator temperature
    jump_threshold: float = 3.0  # jump detection threshold in units of sigma
    gamma: float = 0.5  # kurtosis decay rate for xi
    lambda1: float = 0.1
    lambda2: float = 1e-4
    probabilistic: bool = False
```

Forward pass:
1. `Segmenter` → `segments (B, N, L)`
2. `TCNEncoder` → `Z (B, N, d)`
3. `ConceptProjection` → `A (B, N, K)`
4. `ConceptScorer` on raw segments → `C (B, N, K)` (analytic targets, no grad)
5. `ConceptAttentionPool` → `h (B, K)`, `eta (B, N)`
6. `HorizonDecoder` → `y_hat (B, H)`

`forward` returns a `TCRPOutput` named tuple:
```python
TCRPOutput(y_hat, h, A, C, eta)
```
All fields needed downstream for loss computation and analysis.

---

## Phase 8 — Training
> `tcrp/training/`

### T-14 · Loss functions
**File:** `training/losses.py`  
**Paper:** Eq. 12

Implement `TCRPLoss(lambda1: float, lambda2: float)`:
- `forecast_loss`: `F.mse_loss(y_hat, y_true)`
- `align_loss`: call `alignment_loss(A, C)` from T-09
- `reg_loss`: Frobenius norm of concept projection weight matrix
- `total = forecast_loss + lambda1 * align_loss + lambda2 * reg_loss`
- Return all four terms as a `LossBundle` named tuple for logging

### T-15 · Trainer
**File:** `training/trainer.py`

Implement `Trainer(model, config)`:
- Adam optimiser, lr `1e-3`, weight decay 0 (L2 applied manually via `reg_loss`)
- `ReduceLROnPlateau` scheduler on validation MSE, patience 5, factor 0.5
- Early stopping: patience 10 epochs on validation MSE
- `train_epoch(loader) -> LossBundle`: one epoch, return mean losses
- `validate(loader) -> dict`: return `{mae, mse, align_loss}` on validation set
- `fit(train_loader, val_loader, max_epochs=100)`: full training loop with logging
- Checkpoint: save best model (lowest val MSE) to `checkpoints/best.pt`
- Log to console every epoch: `epoch | train_mse | val_mse | align_loss | lr`

---

## Phase 9 — TCRP Analysis (Relevance Propagation)
> `tcrp/analysis/`

### T-16 · LRP engine
**File:** `analysis/lrp.py`  
**Paper:** Eq. 13 (LRP-ε background), Appendix C

Implement `LRPEngine`:

- `lrp_linear_eps(layer: nn.Linear, a_in, R_out, eps=1e-6) -> Tensor`:
  - Implements Eq. 13: `R_in[i] = sum_j (w[j,i] * a[i] / (z[j] + eps*sign(z[j]))) * R_out[j]`
  - Input: `a_in (B, d_in)`, `R_out (B, d_out)` — returns `R_in (B, d_in)`

- `lrp_gamma_conv(layer: nn.Conv1d, a_in, R_out, gamma=0.25, eps=1e-6) -> Tensor`:
  - Modified weights: `w_gamma = w + gamma * w.clamp(min=0)`
  - Recompute `z` with modified weights; then apply standard LRP-ε rule
  - Must handle causal padding correctly — strip padding before returning

- `lrp_relu(a_in, R_out) -> Tensor`:
  - Pass-through: `R_in = R_out` (relevance passes unchanged through ReLU at explanation time)

- `lrp_mean_pool(a_in: Tensor, R_out: Tensor) -> Tensor`:
  - `a_in (B, d, T_seg)`, `R_out (B, d)` — distribute equally across time
  - `R_in[b, d, t] = R_out[b, d] / T_seg`

### T-17 · TCRP analysis pass
**File:** `analysis/tcrp_analysis.py`  
**Paper:** §5, Eqs. 14–19, Algorithm 1

Implement `TCRPAnalyser(model: TCRPForecaster)`:

`analyse(x: Tensor, h_star: int = 0) -> TCRPExplanation`:

Step 1 — Run forward pass, cache all activations:
- `output = model(x)` with `torch.no_grad()`
- Manually cache: `Z` per segment, `A`, `h`, `eta`, `y_hat`

Step 2 — Initialise relevance at horizon step `h_star`:
- `R_out = y_hat[:, h_star]` — shape `(B,)`

Step 3 — Through decoder (Eq. 14):
- `R_h[k] = Theta[h_star, k] * h[k] / (Theta[h_star] @ h + eps) * R_out`
- Output: `R_h (B, K)` — **concept relevance vector**

Step 4 — Through attention pooling (Eq. 15):
- `R_A[n, k] = eta[n] * R_h[k]`
- Output: `R_A (B, N, K)` — segment × concept relevance matrix

Step 5 — Through concept projection (Eq. 16):
- `R_z[n, d] = sum_k w[k,d] * z[n,d] / (a[n,k] + eps) * R_A[n,k]`
- Output: `R_z (B, N, d)` — back into encoder latent space

Step 6 — Through encoder (layer-by-layer LRP):
- Call `lrp_mean_pool`, then traverse TCN layers in reverse order
- For each `CausalDilatedBlock`: call `lrp_relu` then `lrp_gamma_conv` on each conv
- Output: `R_s (B, N, L)` — per-segment raw-timestep relevance

Step 7 — Temporal assembly (Eq. 17):
- For each `t in 0…T-1`: `R_x[t] = sum_{n: t in s_n} R_s[n, t - t_n] / overlap[t]`
- Use `model.segmenter.start_indices` and `model.segmenter.overlap_counts`
- Output: `R_x (B, T)` — **global temporal relevance map**

Step 8 — Concept-conditional temporal maps (Eq. 18):
- `weight[n, k] = R_A[n, k] / R_A[n].sum(k)` — concept soft weight per segment
- `R_x_k[k, t] = sum_{n: t in s_n} weight[n, k] * R_s[n, t-t_n] / overlap[t]`
- Output: `R_x_cond (B, K, T)` — **K concept-conditional temporal maps**

Return `TCRPExplanation`:
```python
@dataclass
class TCRPExplanation:
    R_h:      Tensor  # (B, K)    concept relevance vector
    R_A:      Tensor  # (B, N, K) segment × concept matrix
    R_x:      Tensor  # (B, T)    global temporal map
    R_x_cond: Tensor  # (B, K, T) concept-conditional maps
    eta:      Tensor  # (B, N)    attention weights
    A:        Tensor  # (B, N, K) concept activations
    C:        Tensor  # (B, N, K) analytic concept scores
```

### T-18 · Conservation check
**Paper:** Theorem 1

Implement `verify_conservation(explanation: TCRPExplanation, y_hat: Tensor, h_star: int, tol=1e-4) -> bool`:
- Assert `|R_x.sum(dim=-1) - y_hat[:, h_star]| < tol` for all samples in batch
- Assert `|R_x_cond.sum(dim=1) - R_x| < tol` (map decomposition, Prop. 2)
- Assert `|R_h.sum(dim=-1) - y_hat[:, h_star]| < tol` (concept decomp., Prop. 1)
- Log any violation with the actual and expected values
- This must pass on every trained model before results are reported

---

## Phase 10 — Data Pipeline
> `tcrp/data/`

### T-19 · Dataset loaders
**File:** `data/datasets.py`

Implement `TimeSeriesDataset(path, split, T, H, normalise=True)`:
- Supports: `ETTh1`, `ETTm2`, `Weather`, `ExchangeRate`, `GEFCOM2014`
- Loads CSV, applies z-score normalisation per channel using train-set statistics only
- Returns `(x, y)` pairs: `x (T,)`, `y (H,)` for univariate; `x (T, V)`, `y (H, V)` for multivariate
- Standard splits: train 60%, val 20%, test 20% (follow TimesNet convention)
- `DataLoader` wrappers: `get_loaders(dataset_name, batch_size=32, num_workers=4)`

### T-20 · Preprocessing utilities
**File:** `data/preprocessing.py`

- `zscore_normalise(train, val, test) -> tuple`: fit on train, transform all three
- `inverse_transform(y_pred, mean, std) -> Tensor`: undo normalisation for metric computation
- `check_no_leakage(train_indices, val_indices, test_indices)`: assert no overlap

---

## Phase 11 — Evaluation
> `tcrp/eval/`

### T-21 · Forecasting metrics
**File:** `eval/metrics.py`

Implement, all operating on unnormalised predictions:
- `mae(y_pred, y_true) -> float`
- `mse(y_pred, y_true) -> float`
- `mase(y_pred, y_true, y_naive) -> float`: MAE relative to seasonal naive baseline
- `scaled_mae(y_pred, y_true, train_std) -> float`: divide by train std (TimesNet convention)
- `scaled_mse(y_pred, y_true, train_std) -> float`
- `evaluate_all(model, loader, inverse_transform_fn) -> dict`: runs full test loop, returns all metrics

### T-22 · Concept Alignment Score
**File:** `eval/metrics.py`  
**Paper:** §7.3

Implement `concept_alignment_score(R_h: Tensor, expert_labels: Tensor) -> dict`:
- `R_h (N_samples, K)` — concept relevance vectors
- `expert_labels (N_samples,)` — integer index of the expert-identified primary concept
- For each sample, `predicted_concept = argmax(|R_h|)`
- `expert_labels` encoding: `0=monotonicity, 1=tendency, 2=stochasticity, 3=periodicity`
- Return `{CAS_mu, CAS_tau, CAS_xi, CAS_rho, CAS_avg}` as per-concept and overall accuracy
- Note: annotators should be instructed to label noise-dominated segments as class 2 (stochasticity)

### T-23 · Temporal faithfulness
**File:** `eval/metrics.py`  
**Paper:** §7.3

Implement `temporal_faithfulness(model, x, y_true, R_x, p=0.2) -> dict`:
- Identify top-`p` fraction of timesteps by `|R_x|`
- **Comprehensiveness**: `original_mse - masked_mse` where masked sets top-p timesteps to channel mean
- **Sufficiency**: `mse` when retaining only the top-p timesteps (rest set to channel mean)
- Return `{comprehensiveness, sufficiency}`

### T-24 · Baseline runners
**File:** `eval/baselines.py`

Thin wrappers calling external model implementations:
- `DLinearBaseline(H)`: from `ts_library.dlinear`
- `NBEATSBaseline(H)`: from `ts_library.nbeats`  
- `PatchTSTBaseline(H)`: from `ts_library.patchtst`
- Each exposes `.fit(train_loader, val_loader)` and `.predict(x) -> Tensor`
- Each stores test predictions for downstream CAS comparison

---

## Phase 12 — Tests

### T-25 · Unit tests
**File:** `tests/test_concepts.py`

- `test_monotonicity_extreme_cases`: all-increasing → `mu_signed ≈ 1`; all-decreasing → `mu_signed ≈ -1`
- `test_curvature_parabola`: convex parabola → `kappa_signed > 0`; concave → `kappa_signed < 0`
- `test_tendency_four_regimes`: one test per row of Table 1 in the paper
- `test_periodicity_pure_sine`: sine at each candidate period → `rho_p ≈ 1` for the matching period
- `test_stochasticity_brownian`: 1000 Brownian samples `torch.cumsum(torch.randn(L), 0)` → `xi.mean() > 0.5`
- `test_stochasticity_linear`: linearly increasing segment → `xi < 0.2` (persistent, H ≈ 1, phi_H ≈ 0)
- `test_stochasticity_sine`: pure sinusoid → `xi < 0.15` (phi_flat ≈ 0 dominates)
- `test_stochasticity_cauchy`: Cauchy-distributed increments → `xi < 0.2` (phi_kurt ≈ 0 dominates)
- `test_stochasticity_geometric_mean_property`: set one sub-score to near-0 → `xi < 0.05` regardless of other two
- `test_stochasticity_constant_segment`: all-constant input → `xi = 0.0` (degenerate handling)
- `test_concept_vector_shape`: output is `(B, 5 + len(periods))` for batched input
- `test_concept_vector_ranges`: all values within stated ranges for 1000 random segments
- `test_concept_names_length`: `len(scorer.concept_names) == K`
- `test_gradients`: `torch.autograd.gradcheck` on the full `ConceptScorer` including `xi` in double precision

**File:** `tests/test_architecture.py`

- `test_segmenter_shapes`: correct `(B, N, L)` output for various `(T, L, stride)` combinations
- `test_segmenter_coverage`: every timestep covered by at least one segment
- `test_encoder_weight_sharing`: same segment at two positions → same encoder output
- `test_bottleneck_shape`: `A` has shape `(B, N, K)`
- `test_alignment_loss_zero`: when `A == C`, loss is 0
- `test_alignment_loss_grad`: loss gradient flows back to encoder weights
- `test_pooling_output`: `h` shape `(B, K)`; `eta` sums to 1 over N
- `test_decoder_linearity`: output is exactly linear in `h`
- `test_full_forward_shapes`: end-to-end forward pass produces correct shapes

**File:** `tests/test_analysis.py`

- `test_conservation_theorem1`: `R_x.sum() ≈ y_hat[h_star]` to within `1e-4`
- `test_prop1_concept_decomp`: `R_h.sum() ≈ y_hat[h_star]`
- `test_prop2_map_decomp`: `R_x_cond.sum(K) ≈ R_x` elementwise
- `test_R_A_structure`: `R_A[n,k] = eta[n] * R_h[k]` exactly
- `test_conditional_weights_sum_to_one`: for each `(b,n,t)`, weights over K sum to 1
- `test_analysis_no_data_leakage`: analysis pass uses only `torch.no_grad()`

---

## Phase 13 — Experiment scripts

### T-26 · Training entry point
**File:** `scripts/train.py`

```
python train.py --dataset ETTh1 --H 96 --lambda1 0.1 --seed 42
```

- Loads config from YAML, merges with CLI args
- Calls `Trainer.fit()`; logs to stdout and `runs/{dataset}/{timestamp}/`
- Saves `best.pt`, `config.yaml`, `train_log.csv`
- Reports final test metrics after training

### T-27 · Ablation runner
**File:** `scripts/ablation.py`

Grid over `lambda1 ∈ [0, 0.01, 0.05, 0.1, 0.2, 0.5]`, `L ∈ [10, 15, 20, 25, 30, 40]`, `hard vs soft concept scores`, and `stochasticity weights (w1, w2, w3) ∈ {equal (1/3,1/3,1/3), Hurst-heavy (0.6,0.2,0.2), flatness-heavy (0.2,0.6,0.2)}`.  
For each configuration: train, evaluate, save metrics to `ablation_results.csv`.

### T-28 · Analysis visualiser
**File:** `scripts/visualise.py`

Given a checkpoint and a look-back window:
1. Run `TCRPAnalyser.analyse()` 
2. Plot 1: `R_x` overlaid on raw `x` (matplotlib, shared x-axis)
3. Plot 2: Stacked area chart of `R_x_cond` — one area per concept, colours matching paper figure
4. Plot 3: `R_h` bar chart — concept relevance with sign
5. Plot 4: `R_A` heatmap — `(N, K)` matrix, rows=segments, cols=concepts
6. Save all four plots to `figures/{run_id}/`

---

## Dependency and ordering

```
T-01 → T-02 → T-04
T-03 → T-04
T-03b → T-04
T-03c → T-04
T-03d → T-04
T-03e → T-04  (internally calls T-01 for slope break)
T-03f → T-04
T-04 → T-09, T-13

T-05 → T-13
T-06 → T-07 → T-13
T-08 → T-13
T-09 → T-14
T-10 → T-13, T-17
T-11 → T-13, T-17
T-12 → (optional, after T-11)

T-13 → T-14, T-17
T-14 → T-15
T-16 → T-17
T-17 → T-18

T-19 → T-20 → T-15, T-24
T-21 → T-26
T-22 → T-26
T-23 → T-26

T-25 (unit tests): run after each phase
T-26 → T-27, T-28
```

Minimum viable path for a first end-to-end run on ETTh1:  
**T-01 → T-02 → T-03 → T-03b → T-03c → T-03d → T-03e → T-03f → T-04 → T-05 → T-06 → T-07 → T-08 → T-09 → T-10 → T-11 → T-13 → T-14 → T-15 → T-19 → T-20 → T-21 → T-26**

Add T-16, T-17, T-18 for the analysis pass.  
Add T-22, T-23 for interpretability evaluation.
