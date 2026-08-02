"""DSM+ — joint denoising score matching + reconstruction (Vincent/Song-style)."""
from __future__ import annotations

import torch
import torch.nn as nn


def dsm_plus_loss(
    model: nn.Module,
    scheduler,
    x_0: torch.Tensor,
    t: torch.Tensor,
    lambda_dsm: float = 0.5,
) -> torch.Tensor:
    """L = MSE(x0_hat, x0) + λ MSE(ε_hat, ε)."""
    x_t, noise = scheduler.q_sample(x_0, t)
    x0_hat, _ = model(x_t, t)
    rec = nn.functional.mse_loss(x0_hat, x_0)
    ab = scheduler.alpha_bar[t].view(-1, 1)
    implied = (x_t - torch.sqrt(ab) * x0_hat) / torch.sqrt(1.0 - ab + 1e-8)
    dsm = nn.functional.mse_loss(implied, noise)
    return rec + float(lambda_dsm) * dsm


def dsm_plus_residual_term(
    x0_hat: torch.Tensor,
    x_0: torch.Tensor,
    x_t: torch.Tensor,
    noise: torch.Tensor,
    t: torch.Tensor,
    scheduler,
    lambda_dsm: float = 0.5,
) -> torch.Tensor:
    """Per-sample additive DSM+ term (already have forward)."""
    rec_ps = nn.functional.mse_loss(x0_hat, x_0, reduction="none").mean(dim=1)
    ab = scheduler.alpha_bar[t].view(-1, 1)
    implied = (x_t - torch.sqrt(ab) * x0_hat) / torch.sqrt(1.0 - ab + 1e-8)
    dsm_ps = nn.functional.mse_loss(implied, noise, reduction="none").mean(dim=1)
    return (1.0 - float(lambda_dsm)) * rec_ps + float(lambda_dsm) * dsm_ps
