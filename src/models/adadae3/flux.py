"""FLUX — light flow-matching velocity head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FluxHead(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, time_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, input_dim),
        )
        self.time_dim = time_dim

    def embed_t(self, t: torch.Tensor) -> torch.Tensor:
        # simple Fourier
        freqs = torch.arange(self.time_dim // 2, device=t.device, dtype=torch.float32) + 1
        a = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(a), torch.cos(a)], dim=1)[:, : self.time_dim]

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        e = self.embed_t(t)
        return self.net(torch.cat([x, e], dim=1))


def flux_loss(head: FluxHead, x0: torch.Tensor, noise: torch.Tensor, t_norm: torch.Tensor) -> torch.Tensor:
    """Conditional flow: x_t = (1-t)x0 + t*noise; target v = noise - x0."""
    xt = (1.0 - t_norm.unsqueeze(1)) * x0 + t_norm.unsqueeze(1) * noise
    v_pred = head(xt, t_norm)
    v_tgt = noise - x0
    return F.mse_loss(v_pred, v_tgt)


@torch.inference_mode()
def flux_residual_score(head: FluxHead, x: torch.Tensor, t_norm: float = 0.5) -> torch.Tensor:
    t = torch.full((x.size(0),), t_norm, device=x.device)
    noise = torch.randn_like(x)
    xt = (1.0 - t_norm) * x + t_norm * noise
    v = head(xt, t)
    # residual magnitude as anomaly score
    s = v.norm(dim=1)
    med = s.median().clamp_min(1e-8)
    return (s / med).float()
