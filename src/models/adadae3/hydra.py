"""HYDRA — shared trunk with light multi-task heads."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class HydraHeads(nn.Module):
    def __init__(self, latent_dim: int, input_dim: int, n_heads: int = 3):
        super().__init__()
        self.recon = nn.Linear(latent_dim, input_dim)
        self.score = nn.Linear(latent_dim, 1)
        self.ssl = nn.Linear(latent_dim, latent_dim)
        self.n_heads = n_heads

    def forward(self, z: torch.Tensor, stop_grad_ssl: bool = True) -> Dict[str, torch.Tensor]:
        out = {
            "recon": self.recon(z),
            "score": self.score(z).squeeze(-1),
        }
        z_ssl = z.detach() if stop_grad_ssl else z
        out["ssl"] = self.ssl(z_ssl)
        return out
