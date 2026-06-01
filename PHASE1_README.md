# Phase 1 — Analytic Concept Scores

Pure differentiable functions and modules for computing temporal concept scores on time series segments.

## Overview

This module implements four analytic concept scoring functions that characterize temporal behavior without learned parameters:

1. **T-01: Soft Monotonicity** (`monotonicity.py`)
2. **T-02: Soft Curvature & Observed Tendency** (`curvature.py`)  
3. **T-03: Periodicity Score** (`periodicity.py`)
4. **T-04: Full Concept Vector** (`concept_vector.py`)

All functions:
- Accept raw numpy arrays or torch tensors
- Return fully differentiable outputs w.r.t. inputs
- Handle both single sequences `(L,)` and batched inputs `(B, L)`

## Module Details

### T-01: Soft Monotonicity Score

**File:** `monotonicity.py`

Measures increasing/decreasing tendency via first differences.

```python
from tcrp.concepts import soft_monotonicity

s = torch.randn(100)  # sequence of length 100
result = soft_monotonicity(s, alpha=5.0)

# Returns MonotonicityScores named tuple:
result.mu        # sigmoid of avg first differences, shape ()
result.mu_signed # range (-1, 1): 1 → increasing, -1 → decreasing, 0 → no trend
result.mu_mag    # absolute magnitude, range (0, 1)
```

**Interpretation:**
- `mu_signed ≈ 1`: monotonically increasing sequence
- `mu_signed ≈ -1`: monotonically decreasing sequence  
- `mu_signed ≈ 0`: no clear monotonic trend (white noise)

**Verification:**
- ✓ Constant-increasing input → `mu_signed ≈ 0.987` (near 1.0)
- ✓ Constant-decreasing input → `mu_signed ≈ -0.987` (near -1.0)
- ✓ White noise → `mu_signed ≈ 0.0`
- ✓ Gradient checking PASSED

### T-02: Soft Curvature Score

**File:** `curvature.py`

Measures concavity/convexity via second differences and computes observed tendency.

```python
from tcrp.concepts import soft_curvature

s = torch.randn(100)
mono = soft_monotonicity(s)

result = soft_curvature(s, mu_signed=mono.mu_signed, beta=5.0)

# Returns CurvatureScores named tuple:
result.kappa_signed  # signed curvature, range (-1, 1)
result.tau           # observed tendency = mu_signed * kappa_signed
```

**Four-Regime Behavior (Table from paper):**

| Pattern | mu_signed | kappa_signed | tau | Interpretation |
|---------|-----------|--------------|-----|----------------|
| Accelerating rise | + | + | + | Rising with increasing rate |
| Decelerating rise | + | − | − | Rising with decreasing rate |
| Accelerating fall | − | − | + | Falling with increasing rate |
| Decelerating fall | − | + | − | Falling with decreasing rate |

**Verification:**
- ✓ Four-regime table verified
- ✓ Batched input handling correct
- ✓ Gradient checking PASSED

### T-03: Periodicity Score

**File:** `periodicity.py`

Detects periodic components via FFT power spectrum analysis.

```python
from tcrp.concepts import periodicity_score

s = torch.randn(256)
periods = [4, 8, 16, 32]

rho = periodicity_score(s, periods=periods)
# rho shape: (len(periods),) or (B, len(periods)) for batches

# Returns power ratios normalized to [0, 1]
# rho_p ≈ 1.0 → strong periodic component at period p
# rho_p ≈ 0.0 → weak periodic component at period p
```

**Verification:**
- ✓ Pure sinusoid at period 16 → `rho_16 ≈ 1.0`
- ✓ White noise → all `rho_p ≈ uniform / bins`
- ✓ Mixed sinusoids → power distributed by amplitude
- ✓ Gradient checking PASSED

### T-04: Full Temporal Concept Vector

**File:** `concept_vector.py`

PyTorch `nn.Module` combining all three concept scores into a single batched operation.

```python
from tcrp.concepts import ConceptScorer

scorer = ConceptScorer(
    alpha=5.0,
    beta=5.0,
    periods=[4, 8, 16, 32]
)

# Batched input
s = torch.randn(16, 128)  # 16 sequences of length 128
c = scorer(s)  # shape: (16, 8)

# Column layout:
# c[:, 0] = mu_signed       (range: -1 to 1)
# c[:, 1] = mu_mag          (range: 0 to 1)
# c[:, 2] = kappa_signed    (range: -1 to 1)
# c[:, 3] = tau             (range: -1 to 1)
# c[:, 4] = rho_p4          (range: 0 to 1)
# c[:, 5] = rho_p8          (range: 0 to 1)
# c[:, 6] = rho_p16         (range: 0 to 1)
# c[:, 7] = rho_p32         (range: 0 to 1)

# Access column names
names = scorer.concept_names
# ['mu_signed', 'mu_mag', 'kappa_signed', 'tau', 'rho_p4', 'rho_p8', 'rho_p16', 'rho_p32']
```

**Verification:**
- ✓ Correct output shape `(B, K)` where K = 4 + len(periods)
- ✓ Column order verified
- ✓ All values in expected ranges
- ✓ Fully differentiable
- ✓ Gradient checking PASSED

## Implementation Highlights

### Batching Support
All functions handle both:
- Single sequences: `(L,)` → outputs shape `()`
- Batched sequences: `(B, L)` → outputs shape `(B,)` or `(B, K)`

### Numerical Stability
- Sigmoid output clamping for safe numerics
- FFT power spectrum normalized with DC exclusion
- Edge case handling (constant sequences, zero variance)

### Differentiability
- All operations use PyTorch differentiable functions
- No in-place operations
- Gradient flow verified via `torch.autograd.gradcheck`

## Testing

Run the comprehensive test notebook:

```bash
cd /home/TCRP
jupyter notebook test_phase1_concepts.ipynb
```

## References

- Paper: TCRP v3 (temporal concept relevance propagation)
- Definition 1: Soft monotonicity (Eq. 1–2)
- Definition 2: Soft curvature (Eq. 3–4)
- Definition 3: Periodicity (Eq. 5)
- Equation 6: Full concept vector composition
- Table 1: Four-regime behavior classification
- Table 2: Concept vector column specifications

## Status

✅ All Phase 1 implementations complete and verified
✅ All tests passing
✅ Gradient checking passed for all modules
✅ Ready for integration into Phase 2 (Model Architecture)
