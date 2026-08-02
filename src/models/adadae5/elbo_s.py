"""ELBO-S — diffusion ELBO / NLL proxy as a scoring view (ε-residual grid).

Prefer reuse of residual already computed in _score_views when available;
standalone grid kept for calibrate/aux paths.
"""
from __future__ import annotations

from typing import List, Optional

import torch


@torch.inference_mode()
def elbo_score_view(
    model,
    scheduler,
    xb: torch.Tensor,
    timesteps: List[int],
    weights: Optional[List[float]] = None,
    score_seed: int = 0,
) -> torch.Tensor:
    """Approximate -log p(x) ∝ Σ_t w_t ||ε - ε̂||^2 along SSTS grid."""
    b = xb.size(0)
    device = xb.device
    acc = torch.zeros(b, device=device, dtype=torch.float32)
    rng = torch.Generator(device=device)
    rng.manual_seed(int(score_seed) + 91)
    if weights is None:
        weights = [1.0 / max(len(timesteps), 1)] * len(timesteps)
    for ti, t_val in enumerate(timesteps):
        w = float(weights[ti])
        t = torch.full((b,), int(t_val), device=device, dtype=torch.long)
        noise = torch.randn(xb.shape, generator=rng, device=device, dtype=xb.dtype)
        ab = scheduler.alpha_bar[t].view(-1, 1)
        x_t = torch.sqrt(ab) * xb + torch.sqrt(1.0 - ab) * noise
        x0_hat, _ = model(x_t, t)
        implied = (x_t - torch.sqrt(ab) * x0_hat) / torch.sqrt(1.0 - ab + 1e-8)
        acc = acc + w * torch.norm(noise - implied, dim=1)
    return acc


def elbo_from_residual(residual_view: torch.Tensor) -> torch.Tensor:
    """Reuse residual accumulator from _score_views as ELBO-S proxy (no extra grid)."""
    return residual_view.float()
