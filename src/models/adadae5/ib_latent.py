"""IB-LATENT — compression pressure on z only (recon already in main loss)."""
from __future__ import annotations

import torch


def ib_latent_loss(
    z: torch.Tensor,
    x_hat: torch.Tensor | None = None,
    x_0: torch.Tensor | None = None,
    beta: float = 0.01,
) -> torch.Tensor:
    """
    Proxy IB compression: β ||z||^2.
    Recon term intentionally omitted — already counted in the main objective.
    """
    del x_hat, x_0
    return float(beta) * (z.float() ** 2).mean()
