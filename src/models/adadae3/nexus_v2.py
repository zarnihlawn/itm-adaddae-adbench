"""NEXUS-v2 — Barlow Twins + VICReg mix with whitening and stop-grad."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def nexus_v2_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    lambda_bt: float = 0.005,
    mu_var: float = 1.0,
    nu_cov: float = 0.04,
) -> torch.Tensor:
    """Barlow Twins redundancy + VICReg variance/covariance; stop-grad on z2."""
    z1 = F.normalize(z1.float(), dim=1)
    z2 = F.normalize(z2.float().detach(), dim=1)
    b, d = z1.shape
    # Whitening-ish: center
    z1c = z1 - z1.mean(dim=0, keepdim=True)
    z2c = z2 - z2.mean(dim=0, keepdim=True)
    c = (z1c.T @ z2c) / max(1, b)
    on = torch.diagonal(c).add_(-1).pow(2).sum()
    off = _off_diagonal(c).pow(2).sum()
    bt = on + lambda_bt * off
    # VICReg variance
    std1 = torch.sqrt(z1c.var(dim=0) + 1e-4)
    var_loss = torch.relu(1.0 - std1).mean() * mu_var
    cov = (z1c.T @ z1c) / max(1, b - 1)
    cov_loss = _off_diagonal(cov).pow(2).mean() * nu_cov
    return bt + var_loss + cov_loss
