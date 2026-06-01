"""
T-03 · Periodicity Score
Paper: Def. 3, Eq. 5

Computes spectral power concentration at specified frequency bins corresponding to periods.
"""

from typing import List
import torch
from torch import Tensor


def periodicity_score(s: Tensor, periods: List[int]) -> Tensor:
    """
    Compute periodicity scores for a sequence at specified periods.
    
    Args:
        s: Input sequence of shape (L,) or (B, L)
        periods: List of periods to analyze (in units of samples)
        
    Returns:
        Tensor of shape () or (B, len(periods)) containing normalized power at each period's bin.
        Each value is in range [0, 1].
    
    Interpretation:
        - rho_p ≈ 1.0: sequence has strong periodic component at period p
        - rho_p ≈ 0.0 or small: sequence has weak periodic component at period p
        - rho_p ≈ 1/(L//2): sequence is white noise (uniform power distribution)
    """
    # Handle both single sequence (L,) and batched (B, L)
    if s.dim() == 1:
        s = s.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    B, L = s.shape
    
    # Compute DFT via real FFT (rfft computes only positive frequencies)
    # Output shape: (B, L//2 + 1)
    X = torch.fft.rfft(s, dim=1)
    
    # Power spectrum: |X[nu]|^2
    P = (X.abs() ** 2)  # shape (B, L//2 + 1)
    
    # Total non-DC power (exclude DC component P[:, 0])
    total_power = P[:, 1:].sum(dim=1, keepdim=True)  # shape (B, 1)
    
    # Avoid division by zero
    total_power = torch.clamp(total_power, min=1e-10)
    
    # Compute power ratio for each period
    rho_list = []
    for p in periods:
        # Find the frequency bin corresponding to period p
        # nu_p = L / p gives the frequency in cycles per length
        # In the DFT, bin nu corresponds to frequency nu * (sampling_rate / L)
        # For period p samples, we want frequency 1/p, so bin nu_p = L / p
        nu_p = L / p
        nu_p_int = int(round(nu_p))
        
        # Clamp to valid DFT range [1, L//2]
        # (bin 0 is DC, bins 1 to L//2 are valid positive frequencies)
        nu_p_int = max(1, min(nu_p_int, L // 2))
        
        # Extract power at this bin
        rho_p = P[:, nu_p_int] / total_power[:, 0]  # shape (B,)
        
        # Clamp to [0, 1] to handle numerical edge cases
        rho_p = torch.clamp(rho_p, 0.0, 1.0)
        
        rho_list.append(rho_p)
    
    # Stack all rho values
    rho = torch.stack(rho_list, dim=1)  # shape (B, len(periods))
    
    # Squeeze batch dimension if input was 1D
    if squeeze_output:
        rho = rho.squeeze(0)
    
    return rho
