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

### T-04 · Full temporal concept vector
**File:** `concepts/concept_vector.py`  
**Paper:** Eq. 6, Table 2

Implement `ConceptScorer(alpha, beta, periods)` as a `nn.Module` (no learned parameters):
- `forward(s: Tensor) -> Tensor` where `s` has shape `(B, L)` for batched segments
- Returns concept vector `c` of shape `(B, K)` where `K = 4 + len(periods)`
- Column order: `[mu_signed, mu_mag, kappa_signed, tau, rho_p1, …, rho_pM]`
- All columns differentiable w.r.t. `s`
- Add `concept_names` property returning a list of string labels for each column
- Verify shapes; verify all values in stated ranges; run `gradcheck` on the full vector

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
    K: int = 6          # number of concepts (4 + len(periods))
    periods: list = field(default_factory=lambda: [24, 168])
    alpha: float = 5.0  # monotonicity temperature
    beta: float = 5.0   # curvature temperature
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
- Return `{CAS_mu, CAS_tau, CAS_rho, CAS_avg}` as per-concept and overall accuracy
- `expert_labels` encoding: `0=monotonicity, 1=tendency, 2=periodicity`

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
- `test_concept_vector_ranges`: all values within stated ranges for 1000 random segments
- `test_gradients`: `torch.autograd.gradcheck` on the full `ConceptScorer` with `double()` precision

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

Grid over `lambda1 ∈ [0, 0.01, 0.05, 0.1, 0.2, 0.5]`, `L ∈ [10, 15, 20, 25, 30, 40]`, and `hard vs soft concept scores`.  
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
**T-01 → T-02 → T-03 → T-04 → T-05 → T-06 → T-07 → T-08 → T-09 → T-10 → T-11 → T-13 → T-14 → T-15 → T-19 → T-20 → T-21 → T-26**

Add T-16, T-17, T-18 for the analysis pass.  
Add T-22, T-23 for interpretability evaluation.
