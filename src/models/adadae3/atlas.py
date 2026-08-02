"""ATLAS — FiLM / AdaLN conditioning helpers (meta-φ + t)."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class FilmGenerator(nn.Module):
    """Generate per-layer (gamma, beta) from conditioning vector."""

    def __init__(self, cond_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim * 2),
        )
        self.out_dim = out_dim
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gb = self.net(cond)
        gamma, beta = gb.chunk(2, dim=-1)
        return 1.0 + gamma, beta


def film(h: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return h * gamma + beta


class MetaPhi(nn.Module):
    """Tiny encoder of dataset summary stats → φ."""

    def __init__(self, in_dim: int = 16, out_dim: int = 16):
        super().__init__()
        self.in_dim = in_dim
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.Tanh())

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        x = stats.float()
        if x.numel() < self.in_dim:
            x = torch.nn.functional.pad(x, (0, self.in_dim - x.numel()))
        return self.net(x[: self.in_dim] if x.dim() == 1 else x[..., : self.in_dim])


def dataset_stats_vector(x: torch.Tensor, omni_phi: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Prefer OMNI phi when provided; else 8-d summary of batch."""
    if omni_phi is not None:
        return omni_phi.float().reshape(-1)
    xf = x.float()
    col_mean = xf.mean(dim=0)
    col_std = xf.std(dim=0).clamp_min(1e-8)
    z = (xf - col_mean) / col_std
    skew = (z**3).mean()
    kurt = (z**4).mean()
    return torch.stack(
        [
            col_mean.mean(),
            col_mean.std().clamp_min(0),
            col_std.mean(),
            col_std.std().clamp_min(0),
            skew,
            kurt,
            torch.tensor(float(x.size(0)), device=x.device).log1p(),
            torch.tensor(float(x.size(1)), device=x.device).log1p(),
        ]
    )
