"""STRATA — multi-scale latent pyramid consistency loss."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def strata_pool(z: torch.Tensor, scales: int = 2) -> list[torch.Tensor]:
    """Average-pool latent features along dim (treat as 1D)."""
    outs = [z]
    cur = z
    for _ in range(max(0, scales - 1)):
        # pool pairs of dims; pad if odd
        if cur.size(1) < 2:
            outs.append(cur)
            continue
        if cur.size(1) % 2 == 1:
            cur = F.pad(cur, (0, 1))
        cur = cur.view(cur.size(0), cur.size(1) // 2, 2).mean(dim=2)
        outs.append(cur)
    return outs


def strata_consistency_loss(z_clean: torch.Tensor, z_noisy: torch.Tensor, scales: int = 2) -> torch.Tensor:
    p1 = strata_pool(z_clean, scales)
    p2 = strata_pool(z_noisy, scales)
    loss = z_clean.new_zeros(())
    for a, b in zip(p1, p2):
        loss = loss + F.mse_loss(a, b)
    return loss / max(1, len(p1))
