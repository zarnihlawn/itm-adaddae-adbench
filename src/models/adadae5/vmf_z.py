"""vMF-Z — hyperspherical latent concentration for normals."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def vmf_normalize(z: torch.Tensor) -> torch.Tensor:
    return F.normalize(z.float(), p=2, dim=1)


def vmf_concentration_loss(z: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    """Encourage normals to concentrate: maximize κ μ·z with μ = mean direction."""
    zn = vmf_normalize(z)
    mu = F.normalize(zn.mean(dim=0, keepdim=True), p=2, dim=1)
    # Negative cosine similarity → minimize
    return float(kappa) * (1.0 - (zn * mu).sum(dim=1).mean())
