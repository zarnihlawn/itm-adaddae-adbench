"""PHASOR — complex / amplitude-gated sinusoidal time embedding."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class PhasorTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0 or dim == 0
        self.dim = dim
        self.amp = nn.Sequential(nn.Linear(1, max(dim, 2)), nn.SiLU(), nn.Linear(max(dim, 2), 1), nn.Softplus())

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if self.dim <= 0:
            return t.new_zeros((t.shape[0], 0))
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if emb.size(1) < self.dim:
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.size(1)))
        amp = self.amp(t.float().unsqueeze(1))
        return emb * amp
