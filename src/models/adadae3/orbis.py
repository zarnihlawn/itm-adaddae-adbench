"""ORBIS — spectral residual scoring view (FFT energy of reconstruction error)."""
from __future__ import annotations

import torch


@torch.inference_mode()
def orbis_score(x: torch.Tensor, x0_hat: torch.Tensor, top_k: int = 8) -> torch.Tensor:
    """Top-k frequency energy of residual (batched over features as 1D signal)."""
    resid = (x - x0_hat).float()
    # rFFT along feature dim
    spec = torch.fft.rfft(resid, dim=1)
    power = (spec.real**2 + spec.imag**2)
    k = max(1, min(top_k, power.size(1)))
    top = torch.topk(power, k=k, dim=1, largest=True).values
    s = top.sum(dim=1)
    med = s.median().clamp_min(1e-8)
    return (s / med).float()
