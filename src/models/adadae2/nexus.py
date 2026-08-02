"""NEXUS: Normality-Expanded Unified Self-Supervision (VICReg-style).

Non-contrastive invariance on lightly augmented train points; used as an
auxiliary loss alongside diffusion reconstruction.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def augment_tabular(x: torch.Tensor, noise_std: float = 0.05, drop_p: float = 0.05) -> torch.Tensor:
    """Light train-only tabular augmentation."""
    noise = torch.randn_like(x) * noise_std
    if drop_p > 0:
        mask = (torch.rand_like(x) > drop_p).float()
        return (x + noise) * mask
    return x + noise


def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    sim_coeff: float = 25.0,
    std_coeff: float = 25.0,
    cov_coeff: float = 1.0,
) -> torch.Tensor:
    """VICReg: invariance + variance + covariance (Barlow-adjacent)."""
    # Invariance
    inv = F.mse_loss(z1, z2)

    def _std(z: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        return torch.mean(F.relu(1.0 - std))

    def _cov(z: torch.Tensor) -> torch.Tensor:
        n, d = z.shape
        z = z - z.mean(dim=0)
        cov = (z.T @ z) / max(n - 1, 1)
        off = cov.fill_diagonal_(0)
        return (off**2).sum() / d

    return sim_coeff * inv + std_coeff * (_std(z1) + _std(z2)) + cov_coeff * (_cov(z1) + _cov(z2))


def nexus_ssl_loss(
    model: nn.Module,
    x_0: torch.Tensor,
    t0: torch.Tensor,
    noise_std: float = 0.05,
    drop_p: float = 0.05,
    sim_coeff: float = 25.0,
) -> torch.Tensor:
    """Encode two augmentations at clean timestep t=1; VICReg on latents."""
    x1 = augment_tabular(x_0, noise_std=noise_std, drop_p=drop_p)
    x2 = augment_tabular(x_0, noise_std=noise_std, drop_p=drop_p)
    _, z1 = model(x1, t0)
    _, z2 = model(x2, t0)
    return vicreg_loss(z1, z2, sim_coeff=sim_coeff)
