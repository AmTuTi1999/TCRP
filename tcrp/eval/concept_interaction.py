"""Higher-order concept interaction analysis for TCRPClassifier.

Three complementary measures capture interactions that the linear decoder hides:

1. concept_correlation_matrix  — Pearson correlation of h vectors across the dataset.
   Captures encoder-level coupling: concepts that co-activate (or anti-correlate)
   because they share the same TCN representation.

2. concept_co_attention         — Mean attention co-localisation: E[Σ_n η_nk₁ · η_nk₂].
   Captures which concept pairs consistently attend to the same segments.

3. conditional_relevance_shift  — For each concept pair (k₁, k₂), split samples by
   whether h_k₂ is above/below its median, then compare the mean relevance
   (θ_c[k₁] · h_k₁) of k₁ across the two groups. A non-zero shift means k₁'s
   contribution to the prediction depends on k₂ — interaction through the encoder.

All functions return plain numpy arrays and work with both TCRPClassifier and
AdversarialTCRPClassifier (the adversarial wrapper's base is used automatically).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader


def _unwrap(model) -> object:
    """Return base TCRPClassifier, stripping adversarial wrapper if present."""
    return getattr(model, "base", model)


@torch.no_grad()
def _collect(
    model, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect h (N,K), eta (N,S), and A (N,S,K) across the full loader."""
    model = _unwrap(model)
    model.eval().to(device)

    h_list, eta_list, A_list = [], [], []
    for x, _ in loader:
        out = model(x.to(device))
        h_list.append(out.h.cpu().numpy())
        eta_list.append(out.eta.cpu().numpy())  # (B, S)
        A_list.append(out.A.cpu().numpy())  # (B, S, K)

    h = np.concatenate(h_list, axis=0)  # (N, K)
    eta = np.concatenate(eta_list, axis=0)  # (N, S)
    A = np.concatenate(A_list, axis=0)  # (N, S, K)
    return h, eta, A


# ── 1. Concept correlation matrix ────────────────────────────────────────────


def concept_correlation_matrix(
    model,
    loader: DataLoader,
    device: torch.device | None = None,
) -> np.ndarray:
    """Pearson correlation of concept activations h across the dataset.

    Returns:
        corr : (K, K) float64 array with values in [-1, 1].
               corr[k₁, k₂] > 0  → concepts co-activate (encoder entanglement).
               corr[k₁, k₂] < 0  → concepts anti-correlate (mutual suppression).
    """
    device = device or torch.device("cpu")
    h, _, _ = _collect(model, loader, device)  # (N, K)
    return np.corrcoef(h.T).astype(np.float64)  # (K, K)


# ── 2. Co-attention matrix ────────────────────────────────────────────────────


def concept_co_attention(
    model,
    loader: DataLoader,
    device: torch.device | None = None,
) -> np.ndarray:
    """Mean attention co-localisation E[Σ_n η_nk₁ · η_nk₂] across the dataset.

    A high value for pair (k₁, k₂) means both concepts consistently attend to
    the same segments — they "fire together" spatially.

    Returns:
        co_attn : (K, K) float64, symmetric, non-negative.
                  Diagonal entries are E[Σ_n η²_nk] (self-attention energy).
    """
    device = device or torch.device("cpu")
    _, eta, A = _collect(model, loader, device)  # eta: (N, S)  A: (N, S, K)

    # Attention-weight the activations: wA_nsk = η_ns · A_nsk
    # co_attn[k₁, k₂] = (1/N) Σ_n Σ_s wA_nsk₁ · wA_nsk₂
    # = (1/N) einsum('nsk,nsl->kl', wA, wA)
    wA = eta[:, :, np.newaxis] * A  # (N, S, K)
    co = np.einsum("nsk,nsl->kl", wA, wA) / len(eta)
    return co.astype(np.float64)  # (K, K)


# ── 3. Conditional relevance shift ───────────────────────────────────────────


def conditional_relevance_shift(
    model,
    loader: DataLoader,
    class_idx: int,
    device: torch.device | None = None,
) -> np.ndarray:
    """Pairwise conditional relevance shift for class `class_idx`.

    For each concept pair (k₁, k₂):
      - Split samples into HIGH group (h_k₂ > median) and LOW group (h_k₂ ≤ median).
      - Compute mean relevance of k₁ in each group: rel_k₁ = θ_c[k₁] · h_k₁.
      - shift[k₁, k₂] = mean_rel_k₁(HIGH) − mean_rel_k₁(LOW).

    A large |shift[k₁, k₂]| means k₁'s contribution to class `class_idx` changes
    substantially depending on whether k₂ is active — a higher-order interaction
    mediated by the shared encoder.

    Returns:
        shift : (K, K) float64.
                shift[k₁, k₂] is the change in k₁'s relevance when k₂ is high.
                Diagonal entries are always 0.
    """
    device = device or torch.device("cpu")
    base = _unwrap(model)
    base.eval().to(device)

    h, _, _ = _collect(base, loader, device)  # (N, K)

    # Decoder weights for this class: (K,)
    theta = base.decoder.linear.weight[class_idx].detach().cpu().numpy()  # (K,)

    # Relevance per sample per concept: (N, K)
    relevance = h * theta[np.newaxis, :]  # θ_c[k] · h_k per sample

    K = h.shape[1]
    shift = np.zeros((K, K), dtype=np.float64)

    medians = np.median(h, axis=0)  # (K,) — one threshold per concept

    for k2 in range(K):
        high_mask = h[:, k2] > medians[k2]
        low_mask = ~high_mask
        if high_mask.sum() == 0 or low_mask.sum() == 0:
            continue
        rel_high = relevance[high_mask].mean(axis=0)  # (K,)
        rel_low = relevance[low_mask].mean(axis=0)  # (K,)
        shift[:, k2] = rel_high - rel_low

    np.fill_diagonal(shift, 0.0)
    return shift


# ── Summary helper ────────────────────────────────────────────────────────────


def top_interactions(
    matrix: np.ndarray,
    concept_names: list[str],
    n: int = 10,
    upper_tri: bool = False,
) -> list[tuple[str, str, float]]:
    """Return the top-n concept pairs by absolute value in a (K, K) matrix.

    Args:
        matrix       : (K, K) interaction matrix (correlation, co-attention, or shift).
        concept_names: List of K concept name strings.
        n            : Number of top pairs to return.
        upper_tri    : When True only iterate j > i, so symmetric matrices yield
                       unique unordered pairs instead of both (A,B) and (B,A).
                       Use False for directional (asymmetric) matrices like shift.

    Returns:
        List of (concept_a, concept_b, value) sorted by |value| descending.
    """
    K = matrix.shape[0]
    entries = []
    for i in range(K):
        j_start = i + 1 if upper_tri else 0
        for j in range(j_start, K):
            if i == j:
                continue
            entries.append((concept_names[i], concept_names[j], float(matrix[i, j])))

    entries.sort(key=lambda t: abs(t[2]), reverse=True)
    return entries[:n]
