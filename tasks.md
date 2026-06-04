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

Implement `TCRPLoss(lambda1: float, lambda2: float, lambda3: float = 0.01)`:
- `forecast_loss`: `F.mse_loss(y_hat, y_true)`
- `align_loss`: call `alignment_loss(A, C)` from T-09
  - **Critical**: `C` must be detached (`C = C.detach()`) before loss computation —
    analytic scores are fixed targets, not differentiable through the concept formulas.
    Assert `C.requires_grad == False` and raise `ValueError` if not.
- `stability_loss`: alignment instability across sub-batches (see below)
  - Split the batch into 4 equal sub-batches; compute `corr_k` for each sub-batch;
    return `lambda3 * sum_k var(corr_k across sub-batches)` — penalises erratic alignment
  - Skip if batch size < 16 (too small to split meaningfully)
- `reg_loss`: Frobenius norm of concept projection weight matrix
- `total = forecast_loss + lambda1 * align_loss + lambda3 * stability_loss + lambda2 * reg_loss`
- Return all five terms as a `LossBundle` named tuple for logging:
  `LossBundle(total, forecast, align, stability, reg)`

### T-15 · Trainer
**File:** `training/trainer.py`

Implement `Trainer(model, config)`:

**Optimiser and scheduler:**
- Adam optimiser, lr `1e-3`, weight decay 0 (L2 applied manually via `reg_loss`)
- `ReduceLROnPlateau` on validation forecast MSE, patience 5, factor 0.5

**Lambda-1 warm-up schedule:**
```python
def lambda1_schedule(epoch: int, warmup: int = 20, target: float = 0.1) -> float:
    return target * min(epoch / warmup, 1.0)
```
During warm-up the model trains as a pure forecaster; alignment pressure is
introduced gradually to avoid the early-epoch gradient oscillation described
in the overfitting analysis. Apply this schedule to `lambda1` at each epoch.

**Dual early stopping** (separate patience counters for each objective):
```python
class DualEarlyStopping:
    def __init__(self, patience_forecast=10, patience_align=15, delta=1e-4):
        ...
    def step(self, val_forecast, val_align) -> bool:
        # Returns True only when BOTH counters exhaust patience simultaneously
        ...
```
Stop training only when both forecast and alignment validation losses have
stagnated. Longer patience for alignment (noisier signal).

**Checkpointing — Pareto front:**
```python
class ParetoCheckpointer:
    def update(self, val_mse, val_align, epoch, model): ...
    def best_forecast(self) -> dict: ...   # lowest val_mse
    def best_align(self) -> dict: ...      # lowest val_align
    def best_balanced(self, alpha=0.5) -> dict: ...  # alpha*mse + (1-alpha)*align
```
Maintain a Pareto front of non-dominated checkpoints across (val_mse, val_align).
A checkpoint is dominated if another is better on both metrics simultaneously.
Save each Pareto-optimal checkpoint to `checkpoints/pareto_e{epoch}.pt`.
At test time, load `best_forecast` for predictive evaluation and `best_align`
for interpretability evaluation.

**Per-epoch validation:**
```python
def validate(loader) -> dict:
    # Returns all of:
    return {
        'mae': ...,
        'mse': ...,                    # primary forecast metric
        'align_loss': ...,             # val alignment loss (separate split)
        'stability_loss': ...,         # alignment variance across sub-batches
        'concept_stds': [...],         # std of each concept dim across val segments
        'n_collapsed': ...,            # number of concept dims with std < 0.01
        'cas_val': ...,                # CAS on annotated val set (if available)
    }
```

**Concept collapse detection** (run every epoch, log always):
```python
def concept_collapse_diagnostic(A_val: Tensor, threshold=0.01) -> dict:
    stds = A_val.std(dim=(0,1))          # (K,) std per concept across B*N segments
    collapsed = (stds < threshold)
    return {
        'collapsed_concepts': collapsed.nonzero().squeeze().tolist(),
        'concept_stds': stds.tolist(),
        'n_collapsed': collapsed.sum().item(),
    }
```
If any concept dimension collapses on validation: log a warning, reduce lambda1
by 20%, and continue. Collapse means the encoder has stopped using that concept
direction — a form of overfitting where the bottleneck becomes degenerate.

**Attention entropy regularisation** (optional, controlled by config flag):
```python
# Add to total loss if config.entropy_reg > 0:
entropy_loss = -config.entropy_reg * (eta * eta.log()).sum(dim=1).mean()
```
Penalises attention weight collapse (all weight on one segment). Prevents the
attention mechanism from memorising a single training-period segment position.
Default: `entropy_reg = 0.0` (disabled); enable if attention weights collapse
to near-deterministic during training.

**Training loop logging** (every epoch to stdout and `runs/{dataset}/{ts}/train_log.csv`):
```
epoch | train_mse | val_mse | train_align | val_align | stability | n_collapsed | lr | lambda1
```

**`fit(train_loader, val_loader, align_val_loader, max_epochs=100)`:**
- `align_val_loader`: a separate loader for alignment validation — shuffled
  segments from the *training period* (not the temporal validation split).
  This separates alignment generalisation from forecast generalisation.
  Alignment loss on this loader measures "does the model align its
  representations to trend concepts?" independently of "does it forecast
  the future well?"

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

## Phase 14 — Overfitting Diagnostics
> `tcrp/diagnostics/`

Seven distinct overfitting mechanisms exist in TCRP, each requiring a
different detection and mitigation strategy. This phase implements tooling
to detect all seven during and after training.

---

### T-29 · Concept direction overfitting monitor
**File:** `diagnostics/concept_overfit.py`

Detects: concept projection weights $\bm W$ overfitting to training-period
concept distributions, causing val alignment to diverge from train alignment.

```python
def concept_direction_overfit_report(
    model: TCRPForecaster,
    train_segments: Tensor,    # (N_train, L) raw segments from training period
    val_segments: Tensor,      # (N_val, L) raw segments from validation period
    concept_scorer: ConceptScorer,
) -> dict:
    """
    Computes per-concept Pearson correlation between learned activations
    and analytic scores on both train and val segments.
    Returns correlation gap per concept: high gap signals direction overfitting.
    """
    with torch.no_grad():
        Z_train = model.encoder(train_segments)
        A_train  = model.bottleneck(Z_train)
        C_train  = concept_scorer(train_segments).detach()

        Z_val   = model.encoder(val_segments)
        A_val    = model.bottleneck(Z_val)
        C_val    = concept_scorer(val_segments).detach()

    report = {}
    for k, name in enumerate(concept_scorer.concept_names):
        corr_train = pearson_corr(A_train[:, k], C_train[:, k]).item()
        corr_val   = pearson_corr(A_val[:, k],   C_val[:, k]).item()
        report[name] = {
            'corr_train': corr_train,
            'corr_val':   corr_val,
            'gap':        corr_train - corr_val,   # >0.15 is a warning
        }
    return report
```

**Thresholds:**
- `gap > 0.15`: warning — concept direction is overfitting to training period
- `gap > 0.30`: critical — stop training and reduce `lambda1` or increase dropout
- `corr_val < 0.3` for any concept: the concept is not generalising at all;
  consider removing it from the bottleneck or increasing `L`

---

### T-30 · Attention memorisation detector
**File:** `diagnostics/attention_overfit.py`

Detects: attention weights $\eta_n$ collapsing to memorised training-period
segment positions, rather than learning generalising concept-based attention.

```python
def attention_memorisation_report(
    model: TCRPForecaster,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> dict:
    """
    Measures:
    1. Attention entropy: H(eta) = -sum_n eta_n * log(eta_n)
       Low entropy = attention collapsed to one or few segments (memorisation risk)
    2. Attention position consistency: do the same segment positions (by index)
       receive high weight across different series in the same split?
       High consistency in training but low in validation = positional memorisation.
    3. Concept-driven vs position-driven attention: regress eta_n on
       (a) concept activation magnitude |A_n| and (b) segment position n.
       If position explains more variance than concept activations,
       the attention is position-memorising.
    """
```

**Implementation:**
- For each batch, compute `H_eta = -(eta * eta.log()).sum(dim=1).mean()`
- Log `H_eta` on train and val every epoch; alert if `H_eta_val < 1.0` (near-deterministic)
- Run position regression: `OLS(eta_n ~ position_n)` and concept regression:
  `OLS(eta_n ~ |A_n|.mean(dim=-1))`; report $R^2$ of each
- **Mitigation**: if position $R^2 >$ concept $R^2$ on validation, enable attention
  entropy regularisation (`config.entropy_reg = 0.05`)

---

### T-31 · Orthogonal direction bypass detector
**File:** `diagnostics/bypass_overfit.py`

Detects: encoder learning to encode training-specific information in directions
orthogonal to all concept directions $\bm w_k$, bypassing the bottleneck's
intended constraint (Overfitting Mechanism 3).

```python
def orthogonal_bypass_report(
    model: TCRPForecaster,
    val_loader: DataLoader,
) -> dict:
    """
    For each validation batch:
    1. Compute encoder output Z (B*N, d)
    2. Project Z onto the concept subspace: Z_concept = W^T (W W^T)^{-1} W Z
    3. Compute residual: Z_residual = Z - Z_concept
    4. Measure what fraction of Z's variance lies in the residual:
       bypass_ratio = var(Z_residual) / var(Z)
    A high bypass_ratio means the encoder is using directions outside
    concept space — these are opaque to the TCRP explanation.
    """
    W = model.bottleneck.weight   # (K, d)
    # Projection onto concept subspace (row space of W)
    WtW_inv = torch.linalg.pinv(W @ W.T)
    P_concept = W.T @ WtW_inv @ W   # (d, d) projection matrix

    bypass_ratios = []
    for x, _ in val_loader:
        with torch.no_grad():
            segs = model.segmenter(x)
            Z = model.encoder(segs).reshape(-1, model.config.d)
            Z_concept  = (P_concept @ Z.T).T
            Z_residual = Z - Z_concept
            ratio = Z_residual.var() / (Z.var() + 1e-8)
            bypass_ratios.append(ratio.item())

    return {
        'bypass_ratio_mean': np.mean(bypass_ratios),
        'bypass_ratio_std':  np.std(bypass_ratios),
        # >0.3 means >30% of encoder variance is outside concept space
        'warning': np.mean(bypass_ratios) > 0.3,
    }
```

**Mitigation if `bypass_ratio > 0.3`:** increase `lambda1` to force more of
the encoder's variance into concept-aligned directions, or add an explicit
orthogonality penalty $\mathcal{L}_\text{orth} = \|(\bm I - \bm P_\text{concept})\bm Z\|_F^2$.

---

### T-32 · Spurious concept-target correlation detector
**File:** `diagnostics/spurious_correlation.py`

Detects: decoder learning spurious associations between concept activations
and forecast targets that hold in training but not validation (Mechanism 5 —
the most dangerous form of TCRP overfitting).

```python
def spurious_correlation_report(
    model: TCRPForecaster,
    train_loader: DataLoader,
    val_loader: DataLoader,
    h_star: int = 0,
) -> dict:
    """
    For each concept k, compute the Pearson correlation between:
    - h_k (pooled concept activation) and y_hat[h_star] (forecast value)
    on both training and validation sets.

    If corr_train[k] >> corr_val[k], concept k has a spurious association
    with the forecast target that does not generalise.
    Also check: for each k, does the decoder weight theta[h_star, k] have
    the same sign as corr_val[k]? Sign mismatch = the decoder learned the
    wrong direction from a spurious training correlation.
    """
    h_train, y_train = collect_h_and_y(model, train_loader, h_star)
    h_val,   y_val   = collect_h_and_y(model, val_loader,   h_star)
    theta = model.decoder.Theta[h_star]   # (K,)

    report = {}
    for k, name in enumerate(model.concept_names):
        corr_tr = pearson_corr(h_train[:, k], y_train).item()
        corr_va = pearson_corr(h_val[:, k],   y_val).item()
        sign_match = (theta[k].item() * corr_va) > 0
        report[name] = {
            'corr_train':  corr_tr,
            'corr_val':    corr_va,
            'gap':         corr_tr - corr_va,
            'decoder_w':   theta[k].item(),
            'sign_match':  sign_match,     # False = spurious association
            'warning':     (corr_tr - corr_va > 0.2) or not sign_match,
        }
    return report
```

**Mitigation:** concepts with `warning=True` should be flagged in the
explanation output. If multiple concepts show spurious correlations,
the training period may be too short or too homogeneous; consider
data augmentation (rolling window expansion) or reducing model capacity.

---

### T-33 · Temperature parameter stability check
**File:** `diagnostics/temperature_overfit.py`

Detects: learned temperature parameters $\alpha$ (monotonicity) and $\beta$
(curvature) taking extreme values that make concept scores brittle
(Overfitting Mechanism 6). Only relevant if temperatures are trained.

```python
def temperature_stability_report(
    concept_scorer: ConceptScorer,
    val_segments: Tensor,
    noise_std: float = 0.01,
    n_trials: int = 50,
) -> dict:
    """
    Measures sensitivity of concept scores to small input perturbations.
    For each trial: add Gaussian noise ~N(0, noise_std) to val_segments,
    recompute concept scores, measure mean absolute change per concept.
    High sensitivity = temperature is too large (brittle concept scores).
    """
    C_clean = concept_scorer(val_segments)
    sensitivities = []
    for _ in range(n_trials):
        noise = torch.randn_like(val_segments) * noise_std
        C_noisy = concept_scorer(val_segments + noise)
        sensitivities.append((C_noisy - C_clean).abs().mean(dim=0))  # (K,)

    mean_sensitivity = torch.stack(sensitivities).mean(dim=0)  # (K,)
    return {
        'mean_sensitivity_per_concept': dict(
            zip(concept_scorer.concept_names, mean_sensitivity.tolist())
        ),
        # >0.1 per unit of noise_std is a warning
        'brittle_concepts': [
            name for name, s in zip(concept_scorer.concept_names,
                                    mean_sensitivity.tolist())
            if s / noise_std > 0.1
        ],
    }
```

**Mitigation:** if `brittle_concepts` is non-empty, fix `alpha` and `beta`
to 5.0 (do not learn them). The paper's default is fixed temperatures;
this check enforces that recommendation.

---

### T-34 · Full overfitting dashboard
**File:** `diagnostics/dashboard.py`

Aggregates all diagnostic reports into a single summary printed after
training and saved to `runs/{dataset}/{ts}/overfit_report.json`.

```python
def overfit_dashboard(
    model: TCRPForecaster,
    concept_scorer: ConceptScorer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    align_val_loader: DataLoader,
    train_segments: Tensor,
    val_segments: Tensor,
    h_star: int = 0,
) -> dict:
    report = {
        'concept_direction':   concept_direction_overfit_report(...),  # T-29
        'attention':           attention_memorisation_report(...),      # T-30
        'orthogonal_bypass':   orthogonal_bypass_report(...),           # T-31
        'spurious_correlation': spurious_correlation_report(...),       # T-32
        'temperature':         temperature_stability_report(...),       # T-33
    }

    # Overall health score: fraction of checks passing
    warnings = sum(
        1 for section in report.values()
        for v in (section.values() if isinstance(section, dict) else [section])
        if isinstance(v, dict) and v.get('warning', False)
    )
    report['n_warnings'] = warnings
    report['health'] = 'good' if warnings == 0 else \
                       'caution' if warnings <= 2 else 'overfitting'

    print(f"\n{'='*50}")
    print(f"TCRP Overfit Dashboard — {report['health'].upper()}")
    print(f"{warnings} warnings across {len(report)-2} diagnostic checks")
    print(f"{'='*50}\n")
    for section, data in report.items():
        if section in ('n_warnings', 'health'):
            continue
        print(f"[{section}]")
        for k, v in data.items():
            if 'warning' in str(k) or 'gap' in str(k):
                flag = ' ⚠' if v else ''
                print(f"  {k}: {v}{flag}")
    return report
```

**Run `overfit_dashboard` at:**
- End of training (always)
- Every 10 epochs during training for early detection
- After loading any checkpoint before reporting test results

---

### T-35 · Overfitting unit tests
**File:** `tests/test_overfit_diagnostics.py`

- `test_detach_assertion`: pass `C` with `requires_grad=True` to `TCRPLoss`;
  assert `ValueError` is raised
- `test_bypass_ratio_ideal`: when `W` spans all of $\R^d$ (full rank K=d),
  bypass ratio should be 0
- `test_bypass_ratio_maximum`: when `W` is rank-1, bypass ratio approaches 1
- `test_spurious_sign_match`: manually set `theta[k]` opposite to `corr_val[k]`;
  assert `sign_match=False` reported
- `test_temperature_brittle`: set `alpha=50` (very high); assert brittle
  monotonicity concept is flagged
- `test_temperature_stable`: set `alpha=5` (default); assert no brittle concepts
  on clean synthetic data
- `test_dashboard_health_good`: all-passing diagnostics → `health='good'`
- `test_dashboard_health_overfit`: inject 3+ failing checks → `health='overfitting'`

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

T-15 → T-29, T-30, T-31, T-32, T-33  (run diagnostics after training)
T-29 → T-34
T-30 → T-34
T-31 → T-34
T-32 → T-34
T-33 → T-34
T-34 → T-26  (dashboard must pass before reporting test results)
T-35 (overfit unit tests): run after T-29–T-34
```

Minimum viable path for a first end-to-end run on ETTh1:  
**T-01 → T-02 → T-03 → T-03b → T-03c → T-03d → T-03e → T-03f → T-04 → T-05 → T-06 → T-07 → T-08 → T-09 → T-10 → T-11 → T-13 → T-14 → T-15 → T-19 → T-20 → T-21 → T-26**

Add T-16, T-17, T-18 for the analysis pass.  
Add T-22, T-23 for interpretability evaluation.  
Add T-29–T-35 for overfitting diagnostics (run after first successful training).

---

## Phase T* — Adversarial Concept Purity (Alternative Training Direction)

> **Status:** alternative to the standard Phase 8 training pipeline.
> Not required for the baseline TCRP results. Pursue after baseline experiments
> confirm the orthogonal bypass overfitting mechanism (T-31 bypass_ratio > 0.3
> or T-29 concept direction gap > 0.2 on any concept family).
>
> **Core motivation:** in standard TCRP training, the encoder $g_\phi$,
> concept projection $\bm W$, and decoder $\Theta$ all minimise jointly.
> The encoder can satisfy both $\mathcal{L}_\text{forecast}$ and
> $\mathcal{L}_\text{align}$ simultaneously by encoding spurious
> training-period patterns into concept directions, as long as those
> patterns correlate with both the target and the analytic scores on
> training data. The minimax game breaks this by making the encoder
> adversarial with respect to alignment: it is forced to find
> representations that the concept projection finds hard to align,
> unless those representations also allow accurate forecasting.
> Only genuine trend features survive this tension.

---

### T*-01 · Gradient Reversal Layer
**File:** `model/adversarial.py`

```python
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, alpha: float) -> Tensor:
        ctx.alpha = alpha
        return x.clone()           # identity in forward pass

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        # Negate and scale gradient flowing into the encoder
        # for the alignment loss path only
        return -ctx.alpha * grad_output, None


class GRLLayer(nn.Module):
    """
    Gradient Reversal Layer (Ganin et al., 2016).
    Placed between encoder output and concept projection.
    Forward: identity.
    Backward (alignment path only): negates gradient by factor alpha.
    Forecast loss gradient is NOT reversed — separate backward path.
    """
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha          # set externally by scheduler each epoch

    def forward(self, x: Tensor) -> Tensor:
        return GradientReversal.apply(x, self.alpha)
```

**Key implementation constraint:** the GRL must sit on the alignment loss
path only. The forecast loss gradient must reach the encoder *without*
reversal. Implement this by computing two separate forward passes through
the bottleneck — one through the GRL for the alignment loss, one directly
for the forecast loss — or by using `retain_graph=True` and separate
`.backward()` calls. Do not use a single `.backward()` on the total loss,
as this would route the forecast gradient through the GRL.

**Correct two-path backward:**
```python
# Forward
z = encoder(segments)           # (B, N, d)

# Path 1: forecast (no GRL)
A_forecast = projection(z)
h, eta = pool(A_forecast)
y_hat = decoder(h)
L_forecast = mse(y_hat, y_true)

# Path 2: alignment (through GRL)
z_grl = grl(z)                  # alpha applied on backward only
A_align = projection(z_grl)
C = concept_scorer(segments).detach()
L_align = alignment_loss(A_align, C)

# Backward: separate, ordered
L_forecast.backward(retain_graph=True)   # encoder gets normal forecast grad
L_align.backward()                       # encoder gets REVERSED alignment grad
optimizer.step()
```

Unit tests:
- `test_grl_forward_identity`: output equals input to machine precision
- `test_grl_backward_negation`: gradient through GRL equals `-alpha * upstream_grad`
- `test_grl_alpha_zero`: alpha=0 → gradient is zero (GRL acts as stop-gradient)
- `test_grl_alpha_one`: alpha=1 → gradient fully negated
- `test_two_path_backward`: verify forecast gradient reaches encoder unchanged;
  verify alignment gradient reaches encoder negated; use `register_hook` on
  encoder parameters to inspect gradient contributions from each path

---

### T*-02 · Alpha schedule
**File:** `model/adversarial.py`

```python
def grl_alpha_schedule(
    epoch: int,
    max_epochs: int,
    warmup_epochs: int = 20,
    alpha_max: float = 1.0,
) -> float:
    """
    Ramps alpha from 0 (no reversal, standard cooperative training)
    to alpha_max (full adversarial pressure) over training.

    Phase 1 (0..warmup_epochs): alpha = 0.
        Encoder trains cooperatively with forecast and alignment losses.
        Both losses decrease; encoder learns useful structure before
        adversarial pressure is applied.

    Phase 2 (warmup_epochs..max_epochs): alpha follows the DANN schedule
        (Ganin et al., 2016):
        p = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
        alpha = alpha_max * (2 / (1 + exp(-10 * p)) - 1)
        This ramps smoothly from 0 to alpha_max.

    Rationale: applying full reversal before warm-up causes the encoder
    to receive contradictory signals before it has learned any useful
    structure, causing divergence. The two-phase schedule ensures the
    encoder first establishes a useful representation, then has its
    concept directions purified by the adversarial pressure.
    """
    if epoch < warmup_epochs:
        return 0.0
    p = (epoch - warmup_epochs) / max(max_epochs - warmup_epochs, 1)
    return alpha_max * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)
```

Unit tests:
- `test_schedule_warmup_zero`: alpha == 0 for all epochs < warmup_epochs
- `test_schedule_monotone`: alpha is non-decreasing after warmup
- `test_schedule_max`: alpha approaches alpha_max as epoch → max_epochs
- `test_schedule_continuity`: no discontinuity at warmup boundary

---

### T*-03 · Adversarial TCRP model wrapper
**File:** `model/adversarial.py`

```python
class AdversarialTCRPForecaster(nn.Module):
    """
    Wraps TCRPForecaster with a GRL between encoder and concept projection.
    Exposes the same interface as TCRPForecaster so all downstream
    analysis (T-17 TCRP analysis pass, T-29–T-35 diagnostics) can
    be applied without modification.

    The only architectural addition is the GRLLayer sitting between
    model.encoder and model.bottleneck on the alignment backward path.
    """
    def __init__(self, base_model: TCRPForecaster, alpha: float = 0.0):
        super().__init__()
        self.base   = base_model
        self.grl    = GRLLayer(alpha=alpha)

    def set_alpha(self, alpha: float):
        self.grl.alpha = alpha

    def forward(self, x: Tensor) -> tuple[TCRPOutput, Tensor]:
        """
        Returns (forecast_output, A_align) where:
        - forecast_output: standard TCRPOutput using z directly (no GRL)
        - A_align: concept activations computed through GRL for alignment loss

        Caller uses forecast_output.y_hat for L_forecast
        and A_align for L_align — separate backward paths.
        """
        segs = self.base.segmenter(x)
        z    = self.base.encoder(segs)

        # Forecast path (no GRL)
        A_f   = self.base.bottleneck(z)
        h, eta = self.base.pool(A_f)
        y_hat  = self.base.decoder(h)

        # Alignment path (through GRL — gradient reversed on backward)
        z_grl  = self.grl(z)
        A_align = self.base.bottleneck(z_grl)

        C = self.base.concept_scorer(segs).detach()

        forecast_output = TCRPOutput(y_hat=y_hat, h=h, A=A_f, C=C, eta=eta)
        return forecast_output, A_align
```

**Important:** `self.base.bottleneck` is called twice — once per path.
This is intentional: the two paths are deliberately separate in the
computational graph so that their gradients do not mix.
The projection weights $\bm W$ receive gradients from both paths:
from $\mathcal{L}_\text{forecast}$ (minimise), and from $\mathcal{L}_\text{align}$
through the GRL (also minimise — the GRL only reverses the gradient
flowing into the encoder, not into $\bm W$ itself).

---

### T*-04 · Adversarial trainer
**File:** `training/adversarial_trainer.py`

Extends `Trainer` (T-15) with the two-path backward and alpha scheduling.

```python
class AdversarialTrainer(Trainer):
    """
    Replaces the standard single-loss backward with a two-path backward:
    Path 1: L_forecast → decoder, W, encoder (normal gradient)
    Path 2: L_align → W, GRL → encoder (reversed gradient)

    All monitoring from T-15 (dual early stopping, Pareto checkpointing,
    concept collapse detection, attention entropy) applies unchanged.
    Additional monitoring specific to the adversarial regime:
    - alpha value logged every epoch
    - encoder gradient norms from each path logged separately
    - concept purity score (T*-05) logged every 5 epochs
    """

    def train_step(self, x: Tensor, y_true: Tensor) -> LossBundle:
        self.optimizer.zero_grad()

        forecast_output, A_align = self.model(x)
        C = forecast_output.C   # already detached

        # Loss computation
        L_fc   = F.mse_loss(forecast_output.y_hat, y_true)
        L_al   = alignment_loss(A_align, C)
        L_stab = stability_loss(A_align, C)
        L_reg  = self.config.lambda2 * self.model.base.bottleneck.weight.norm('fro')

        # Two-path backward
        # Path 1: forecast — normal gradient to encoder
        L_fc.backward(retain_graph=True)

        # Path 2: alignment — reversed gradient to encoder via GRL
        (self.config.lambda1 * L_al + self.config.lambda3 * L_stab).backward()

        # Clip gradients before step
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        total = L_fc + self.config.lambda1 * L_al + L_reg
        return LossBundle(total, L_fc, L_al, L_stab, L_reg)

    def fit(self, train_loader, val_loader, align_val_loader, max_epochs=100):
        for epoch in range(max_epochs):
            # Update alpha on schedule
            alpha = grl_alpha_schedule(
                epoch, max_epochs,
                warmup_epochs=self.config.warmup_epochs,
                alpha_max=self.config.alpha_max,
            )
            self.model.set_alpha(alpha)

            # Standard epoch loop from T-15 (reused)
            train_bundle = self.train_epoch(train_loader)
            val_metrics  = self.validate(val_loader, align_val_loader)

            self.log(epoch, alpha, train_bundle, val_metrics)
            self.pareto_checkpointer.update(
                val_metrics['mse'], val_metrics['align_loss'],
                epoch, self.model
            )
            if self.dual_stopper.step(
                val_metrics['mse'], val_metrics['align_loss']
            ):
                print(f"Early stop at epoch {epoch}")
                break
```

New config fields added to `TCRPConfig`:
```python
adversarial:    bool  = False   # enable adversarial training
alpha_max:      float = 1.0     # maximum GRL reversal strength
warmup_epochs:  int   = 20      # epochs before GRL activates
```

---

### T*-05 · Concept purity diagnostic
**File:** `diagnostics/concept_purity.py`

Measures whether learned concept directions $\bm w_k$ are aligned with
the genuine analytic gradient or contaminated by spurious encoder features.
This is the key diagnostic that distinguishes whether the minimax game
has converged to a pure equilibrium.

```python
def concept_purity_score(
    model: AdversarialTCRPForecaster,
    concept_scorer: ConceptScorer,
    segments: Tensor,    # (N, L) — use validation segments
    k: int,
) -> dict:
    """
    Computes cosine similarity between:
    (a) w_k: the learned concept direction in encoder space
    (b) mean_grad_k: the mean gradient of analytic score c_k
        with respect to encoder output z, over the segment batch.

    High cosine similarity → w_k points in the genuine concept direction.
    Low cosine similarity → w_k has been contaminated by spurious features.

    Interpretation:
    > 0.7  : pure — concept direction is genuinely aligned
    0.5–0.7: moderate contamination — monitor closely
    0.3–0.5: significant contamination — increase alpha_max or lambda1
    < 0.3  : severe contamination — adversarial training not converging
    """
    segments = segments.requires_grad_(True)
    z  = model.base.encoder(segments)        # (N, d)
    c  = concept_scorer(segments)[:, k]      # (N,) analytic score

    # dc_k / dz via autograd (chain rule through encoder)
    grad_c_z = torch.autograd.grad(
        outputs=c.sum(),
        inputs=z,
        create_graph=False,
        retain_graph=False,
    )[0]                                     # (N, d)
    mean_grad = F.normalize(
        grad_c_z.mean(dim=0, keepdim=True), dim=1
    )                                        # (1, d) unit vector

    w_k = F.normalize(
        model.base.bottleneck.weight[k].unsqueeze(0), dim=1
    )                                        # (1, d) unit vector

    cosine_sim = (w_k * mean_grad).sum().item()

    return {
        'concept':    concept_scorer.concept_names[k],
        'cosine_sim': cosine_sim,
        'pure':       cosine_sim > 0.7,
        'warning':    cosine_sim < 0.5,
    }


def concept_purity_report(
    model: AdversarialTCRPForecaster,
    concept_scorer: ConceptScorer,
    train_segments: Tensor,
    val_segments: Tensor,
) -> dict:
    """
    Runs concept_purity_score for all K concepts on both train and val segments.
    Reports purity gap (train - val cosine similarity) per concept.
    A positive gap indicates spurious contamination persisting despite
    adversarial training.
    """
    report = {}
    for k in range(len(concept_scorer.concept_names)):
        tr = concept_purity_score(model, concept_scorer, train_segments, k)
        va = concept_purity_score(model, concept_scorer, val_segments,   k)
        report[concept_scorer.concept_names[k]] = {
            'cosine_train': tr['cosine_sim'],
            'cosine_val':   va['cosine_sim'],
            'purity_gap':   tr['cosine_sim'] - va['cosine_sim'],
            'pure_val':     va['pure'],
            'warning':      va['warning'] or (tr['cosine_sim'] - va['cosine_sim'] > 0.15),
        }
    return report
```

Run `concept_purity_report` every 5 epochs and log to
`runs/{dataset}/{ts}/purity_log.csv`. Plot train vs val cosine similarity
per concept over training epochs — this is the primary convergence
diagnostic for the adversarial regime.

---

### T*-06 · Adversarial vs standard comparison experiment
**File:** `scripts/adversarial_compare.py`

```
python adversarial_compare.py --dataset ETTh1 --H 96 --seed 42
```

Trains two models on identical data and seeds:
1. Standard TCRP (T-15 trainer, `adversarial=False`)
2. Adversarial TCRP (T*-04 trainer, `adversarial=True`, `alpha_max=1.0`)

Reports for each:
- Val MSE, MAE (forecast quality)
- CAS per concept family (alignment quality)
- Bypass ratio from T-31 (orthogonal leakage)
- Mean concept purity score from T*-05 (contamination level)
- Concept direction gap from T-29 (train vs val alignment divergence)

Primary success criterion for adversarial training:
- Purity scores higher (less contamination)
- Bypass ratio lower (less orthogonal leakage)
- CAS equal or higher (alignment not degraded)
- Val MSE equal or lower (forecast not degraded)

If adversarial training degrades val MSE while improving purity, report
the Pareto trade-off: the user may accept a small MSE increase for
substantially purer concept directions depending on deployment context.

---

### T*-07 · Adversarial unit tests
**File:** `tests/test_adversarial.py`

- `test_grl_paths_independent`: verify via `register_hook` that the
  forecast loss gradient reaching encoder params does not pass through GRL;
  verify alignment loss gradient reaching encoder params does pass through GRL
- `test_encoder_gradient_signs`: with alpha=1, encoder param gradients from
  alignment path should have opposite sign to encoder param gradients from
  alignment loss computed without GRL
- `test_purity_improves_with_adversarial`: on synthetic data with a known
  spurious correlation injected into the training set, run both standard
  and adversarial training; assert adversarial produces higher purity scores
- `test_purity_score_range`: cosine similarity always in [-1, 1]
- `test_alpha_schedule_integration`: full training loop with adversarial
  trainer on tiny synthetic dataset; assert alpha increases monotonically
  after warmup; assert training does not diverge (loss finite at all epochs)
- `test_two_path_backward_no_mixing`: manually set encoder gradients to
  zero between the two `.backward()` calls; assert Path 1 and Path 2
  gradient contributions are additive and independent

---

## T* Dependency

```
T-13 (base TCRP model) → T*-01 → T*-02 → T*-03 → T*-04
T-14 (loss functions)  → T*-04
T-15 (standard trainer) → T*-04  (adversarial trainer inherits from standard)
T*-03 → T*-05
T*-04 → T*-06
T*-05 → T*-06
T-29–T-34 (overfit diagnostics) → T*-06  (run both diagnostic suites for comparison)
T*-01–T*-05 → T*-07  (unit tests)
```

**T* is independent of T-16–T-28.**  The TCRP analysis pass, evaluation
metrics, and experiment scripts all operate on the trained model weights
and are architecture-agnostic.  Once `AdversarialTCRPForecaster` exposes
the same interface as `TCRPForecaster`, all downstream tasks apply unchanged.

**Recommended sequencing:**
1. Complete baseline training (T-26) and run diagnostics (T-29–T-34)
2. If bypass_ratio > 0.3 or concept direction gap > 0.2: proceed with T*
3. Run T*-06 comparison experiment
4. If adversarial training improves purity without MSE degradation: adopt
   as the default training mode and update T-26 accordingly