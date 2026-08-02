"""RIDGE — Huber/Cauchy robust reconstruction loss (keep DSM+ separately)."""
from __future__ import annotations

import torch


def ridge_huber_loss(
    x_hat: torch.Tensor,
    x_0: torch.Tensor,
    delta: float = 1.0,
    reduction: str = "none",
) -> torch.Tensor:
    """Per-sample mean Huber over features."""
    err = (x_hat.float() - x_0.float()).abs()
    d = float(delta)
    huber = torch.where(err <= d, 0.5 * err**2, d * (err - 0.5 * d))
    per = huber.mean(dim=1)
    if reduction == "mean":
        return per.mean()
    return per
