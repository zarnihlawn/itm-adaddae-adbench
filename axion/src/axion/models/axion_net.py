"""AXION neural modules: FX-Enc + HPD (heteroscedastic head)."""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class ResMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class FXEncoder(nn.Module):
    """Feature-masked residual MLP encoder (SCALE picks depth/width)."""

    def __init__(
        self,
        d_in: int,
        hidden: int = 256,
        latent: int = 64,
        depth: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(d_in * 2, hidden),  # x ⊙ (1-m)  ||  m
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.blocks = nn.ModuleList([ResMLPBlock(hidden, dropout) for _ in range(depth)])
        self.to_latent = nn.Linear(hidden, latent)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, D), mask: (B, D) with 1 = masked (to predict), 0 = visible.
        Returns h (B, H), z (B, L).
        """
        visible = x * (1.0 - mask)
        inp = torch.cat([visible, mask], dim=-1)
        h = self.in_proj(inp)
        for blk in self.blocks:
            h = blk(h)
        z = self.to_latent(h)
        return h, z


class HPDHead(nn.Module):
    """Heteroscedastic Predictive head: μ and log σ² for masked cells."""

    def __init__(self, hidden: int, d_out: int):
        super().__init__()
        self.mu = nn.Linear(hidden, d_out)
        self.log_var = nn.Linear(hidden, d_out)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.mu(h), self.log_var(h)


class AxionNet(nn.Module):
    def __init__(
        self,
        d_in: int,
        hidden: int = 256,
        latent: int = 64,
        depth: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = FXEncoder(d_in, hidden, latent, depth, dropout)
        self.head = HPDHead(hidden, d_in)
        self.d_in = d_in
        self.latent = latent

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, z = self.encoder(x, mask)
        mu, log_var = self.head(h)
        return mu, log_var, z


def gaussian_nll(
    x: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    mask: torch.Tensor,
    log_var_clamp: Tuple[float, float] = (-8.0, 8.0),
) -> torch.Tensor:
    """Mean NLL over masked positions (mask=1)."""
    log_var = torch.clamp(log_var, log_var_clamp[0], log_var_clamp[1])
    # NLL = 0.5 * (log_var + (x-mu)^2 / exp(log_var))
    nll = 0.5 * (log_var + (x - mu) ** 2 / torch.exp(log_var))
    masked = nll * mask
    denom = mask.sum(dim=-1).clamp_min(1.0)
    return masked.sum(dim=-1) / denom
