"""AETHER: Adaptive Energy-Timed Hybrid Scoring.

DSM / epsilon-prediction consistency at train time; path-integral energy view
along SSTS timesteps at inference.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


def dsm_energy_loss(
    model: nn.Module,
    scheduler,
    x_0: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Denoising score matching / epsilon prediction MSE."""
    x_t, noise = scheduler.q_sample(x_0, t)
    x0_hat, _ = model(x_t, t)
    ab = scheduler.alpha_bar[t].view(-1, 1)
    # Implied noise from x0 prediction
    implied = (x_t - torch.sqrt(ab) * x0_hat) / torch.sqrt(1.0 - ab + 1e-8)
    return nn.functional.mse_loss(implied, noise)


@torch.inference_mode()
def aether_path_energy(
    model: nn.Module,
    scheduler,
    xb: torch.Tensor,
    timesteps: List[int],
    weights: Optional[List[float]] = None,
    score_seed: int = 0,
) -> torch.Tensor:
    """Path-integral of ||eps - eps_hat|| along selected timesteps."""
    b = xb.size(0)
    device = xb.device
    acc = torch.zeros(b, device=device, dtype=torch.float32)
    ws = weights if weights is not None else [1.0] * len(timesteps)
    wsum = sum(ws) + 1e-12
    rng = torch.Generator(device=device)
    rng.manual_seed(int(score_seed) + 41)
    for ti, t_val in enumerate(timesteps):
        w = float(ws[ti]) / wsum
        t = torch.full((b,), int(t_val), device=device, dtype=torch.long)
        noise = torch.randn(xb.shape, generator=rng, device=device, dtype=xb.dtype)
        ab = scheduler.alpha_bar[t].view(-1, 1)
        x_t = torch.sqrt(ab) * xb + torch.sqrt(1.0 - ab) * noise
        x0_hat, _ = model(x_t, t)
        implied = (x_t - torch.sqrt(ab) * x0_hat) / torch.sqrt(1.0 - ab + 1e-8)
        acc = acc + w * torch.norm(noise - implied, dim=1)
    return acc
