"""SPIRAL — one-step reverse / path consistency score."""
from __future__ import annotations

from typing import Optional, Sequence

import torch


@torch.inference_mode()
def spiral_consistency_score(
    model: torch.nn.Module,
    scheduler,
    x: torch.Tensor,
    t_frac: float = 0.5,
    score_seed: int = 0,
) -> torch.Tensor:
    """
    Corrupt to mid-t, predict x0_hat, re-noise with same ε proxy via model residual.
    Score = ||x - x0_hat|| + ||ε - ε_implied|| (consistency).
    """
    b = x.size(0)
    T = int(scheduler.num_timesteps)
    t_val = max(1, min(T - 1, int(round(t_frac * (T - 1)))))
    t = torch.full((b,), t_val, device=x.device, dtype=torch.long)
    rng = torch.Generator(device=x.device)
    rng.manual_seed(int(score_seed) + 91)
    noise = torch.randn(x.shape, generator=rng, device=x.device, dtype=x.dtype)
    ab = scheduler.alpha_bar[t.clamp(0, scheduler.alpha_bar.numel() - 1)].view(-1, 1)
    x_t = torch.sqrt(ab) * x + torch.sqrt(1.0 - ab) * noise
    x0_hat, _ = model(x_t, t)
    rec = torch.norm(x - x0_hat, dim=1)
    implied = (x_t - torch.sqrt(ab) * x0_hat) / torch.sqrt(1.0 - ab + 1e-8)
    res = torch.norm(noise - implied, dim=1)
    return torch.nan_to_num(rec + 0.5 * res, nan=0.0, posinf=0.0, neginf=0.0)
