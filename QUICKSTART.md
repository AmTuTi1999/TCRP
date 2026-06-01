# Quick Start Guide — Phase 1 Analytic Concept Scores

## Installation & Import

```python
import sys
sys.path.insert(0, '/home/TCRP')

import torch
from tcrp.concepts import (
    soft_monotonicity,
    soft_curvature,
    periodicity_score,
    ConceptScorer,
)
```

## Basic Usage Examples

### Example 1: Analyze a Single Sequence

```python
import torch

# Create a test sequence
sequence = torch.sin(torch.arange(100, dtype=torch.float32) * 0.1)

# Compute monotonicity
mono = soft_monotonicity(sequence)
print(f"Monotonicity: mu_signed={mono.mu_signed:.4f}, mu_mag={mono.mu_mag:.4f}")
# Output: Monotonicity: mu_signed=-0.0113, mu_mag=0.0113

# Compute curvature
curv = soft_curvature(sequence, mu_signed=mono.mu_signed)
print(f"Curvature: kappa={curv.kappa_signed:.4f}, tau={curv.tau:.4f}")

# Compute periodicity
rho = periodicity_score(sequence, periods=[8, 16, 32])
print(f"Periodicity scores: {rho}")
# Output: tensor([0.0000, 0.0000, 0.0000])
```

### Example 2: Batch Processing with ConceptScorer

```python
# Initialize the concept scorer
scorer = ConceptScorer(alpha=5.0, beta=5.0, periods=[4, 8, 16, 32])

# Create a batch of sequences
batch_size = 16
seq_length = 128
sequences = torch.randn(batch_size, seq_length)

# Compute concept vectors
concept_vectors = scorer(sequences)  # Shape: (16, 8)

# Access individual concepts
for i, name in enumerate(scorer.concept_names):
    col = concept_vectors[:, i]
    print(f"{name:15s}: min={col.min():.4f}, mean={col.mean():.4f}, max={col.max():.4f}")
```

### Example 3: Analyzing Different Temporal Patterns

```python
import torch
import numpy as np

scorer = ConceptScorer(periods=[8, 16, 32])

# Test various patterns
patterns = {
    'Step Up': torch.cat([torch.ones(50), torch.ones(50) * 10]),
    'Step Down': torch.cat([torch.ones(50) * 10, torch.ones(50)]),
    'Sine (P=16)': torch.sin(torch.arange(100) * 2 * np.pi / 16),
    'Noise': torch.randn(100),
}

# Batch them together
s = torch.stack(list(patterns.values()))
c = scorer(s)

# Print results
print(f"{'Pattern':<15} {'mu_signed':>10} {'tau':>10} {'rho_16':>10}")
print("-" * 50)
for i, (name, _) in enumerate(patterns.items()):
    print(f"{name:<15} {c[i,0]:>10.4f} {c[i,3]:>10.4f} {c[i,5]:>10.4f}")
```

### Example 4: Integration with PyTorch Training

```python
import torch.nn as nn
import torch.optim as optim

# Use ConceptScorer as a feature extractor in a model
class TemporalAnalyzer(nn.Module):
    def __init__(self, num_periods=4):
        super().__init__()
        self.scorer = ConceptScorer(periods=[8, 16, 32])
        
        # Output dimension: 4 (monotonicity/curvature) + num_periods
        self.num_concepts = 4 + len([8, 16, 32])
        
        # Optional: downstream layers
        self.classifier = nn.Sequential(
            nn.Linear(self.num_concepts, 64),
            nn.ReLU(),
            nn.Linear(64, 10),  # 10 classes
        )
    
    def forward(self, sequences):
        # Extract concepts (B, L) -> (B, K)
        concepts = self.scorer(sequences)
        
        # Pass through classifier
        logits = self.classifier(concepts)
        return logits

# Training loop
model = TemporalAnalyzer()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Example batch
sequences = torch.randn(32, 128)  # 32 sequences of length 128
targets = torch.randint(0, 10, (32,))

# Forward pass
outputs = model(sequences)
loss = criterion(outputs, targets)

# Backward pass
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## API Reference

### `soft_monotonicity(s: Tensor, alpha: float = 5.0) -> MonotonicityScores`

Computes soft monotonicity score.

**Args:**
- `s`: Input tensor, shape `(L,)` or `(B, L)`
- `alpha`: Temperature parameter for sigmoid (default: 5.0)

**Returns:** `MonotonicityScores` named tuple
- `mu`: Sigmoid of average first differences
- `mu_signed`: Range (-1, 1)
- `mu_mag`: Magnitude, range (0, 1)

---

### `soft_curvature(s: Tensor, mu_signed: Tensor, beta: float = 5.0) -> CurvatureScores`

Computes soft curvature and observed tendency.

**Args:**
- `s`: Input tensor, shape `(L,)` or `(B, L)`
- `mu_signed`: Pre-computed signed monotonicity
- `beta`: Temperature parameter for sigmoid (default: 5.0)

**Returns:** `CurvatureScores` named tuple
- `kappa_signed`: Signed curvature, range (-1, 1)
- `tau`: Observed tendency, range (-1, 1)

---

### `periodicity_score(s: Tensor, periods: List[int]) -> Tensor`

Detects periodic components via FFT.

**Args:**
- `s`: Input tensor, shape `(L,)` or `(B, L)`
- `periods`: List of periods to analyze

**Returns:** Tensor of normalized power ratios
- Shape: `(len(periods),)` or `(B, len(periods))`
- Values: Range [0, 1]

---

### `ConceptScorer(alpha, beta, periods)`

Full concept vector computation module.

**Args:**
- `alpha`: Monotonicity temperature (default: 5.0)
- `beta`: Curvature temperature (default: 5.0)
- `periods`: Periodicity periods (default: [2, 4, 8])

**Methods:**

#### `forward(s: Tensor) -> Tensor`
Compute concept vector for batched sequences.
- **Input:** `(B, L)` tensor
- **Output:** `(B, K)` tensor where K = 4 + len(periods)

#### `concept_names` (property)
List of column labels for concept vector.

**Example:**
```python
scorer = ConceptScorer(periods=[4, 8, 16])
names = scorer.concept_names
# ['mu_signed', 'mu_mag', 'kappa_signed', 'tau', 'rho_p4', 'rho_p8', 'rho_p16']
```

## Interpretation Guide

### Monotonicity (`mu_signed`)
- **+1**: Strictly increasing sequence
- **0**: No clear trend (random walk)
- **-1**: Strictly decreasing sequence

### Magnitude (`mu_mag`)
- **0**: Very weak monotonic trend
- **1**: Very strong monotonic trend

### Curvature (`kappa_signed`)
- **+1**: Concave (accelerating)
- **0**: Linear trend
- **-1**: Convex (decelerating)

### Tendency (`tau`)
Interaction of monotonicity and curvature:
- **Rising + Accelerating** → `tau > 0`
- **Rising + Decelerating** → `tau < 0`
- **Falling + Accelerating** → `tau > 0`
- **Falling + Decelerating** → `tau < 0`

### Periodicity (`rho_p`)
- **≈ 1.0**: Strong periodic component at period p
- **≈ 0.0**: No periodic component at period p
- **≈ 1/(L//2)**: White noise (uniform distribution)

## Performance Notes

- Monotonicity: O(L)
- Curvature: O(L)
- Periodicity: O(L log L) due to FFT
- Full concept vector: O(L log L)

All operations are GPU-compatible (CUDA-enabled if available).

## Troubleshooting

### Gradient issues
All outputs are differentiable. If gradients are None:
```python
s = torch.randn(100, requires_grad=True)
c = scorer(s)
if c.requires_grad:
    loss = c.sum()
    loss.backward()
```

### NaN values
- Check input for extreme values
- Ensure input is float32 or float64
- Use `torch.clamp` if needed

### Shape mismatches
- Single sequences: input `(L,)` → output scalars
- Batches: input `(B, L)` → output `(B,)` or `(B, K)`

## References

- **Paper:** TCRP v3 (Temporal Concept Relevance Propagation)
- **Test Notebook:** `/home/TCRP/test_phase1_concepts.ipynb`
- **Documentation:** `/home/TCRP/PHASE1_README.md`
