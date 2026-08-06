"""MCB — Mask Curriculum Bank for AXION."""
from __future__ import annotations

from typing import Optional, Sequence

import torch


def sample_masks(
    batch_size: int,
    d: int,
    *,
    rates: Sequence[float] = (0.15, 0.3, 0.5),
    block_prob: float = 0.25,
    max_block_frac: float = 0.25,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Sample binary masks on CPU (1 = masked), then move to ``device``."""
    out_device = device or torch.device("cpu")
    cpu = torch.device("cpu")
    rate_idx = torch.randint(0, len(rates), (batch_size,), generator=generator, device=cpu)
    rates_t = torch.tensor(list(rates), dtype=torch.float32, device=cpu)
    p = rates_t[rate_idx].unsqueeze(1).expand(batch_size, d)
    bern = torch.rand(batch_size, d, generator=generator, device=cpu) < p
    mask = bern.float()

    if block_prob > 0 and d >= 4:
        use_block = torch.rand(batch_size, generator=generator, device=cpu) < block_prob
        for i in range(batch_size):
            if not bool(use_block[i]):
                continue
            blk = max(
                1,
                int(d * float(max_block_frac) * float(torch.rand(1, generator=generator, device=cpu).item())),
            )
            blk = min(blk, d)
            start = int(torch.randint(0, d - blk + 1, (1,), generator=generator, device=cpu).item())
            mask[i, :] = 0.0
            mask[i, start : start + blk] = 1.0

    if d > 1:
        for i in range(batch_size):
            if mask[i].sum() < 1:
                j = int(torch.randint(0, d, (1,), generator=generator, device=cpu).item())
                mask[i, j] = 1.0
            if mask[i].sum() >= d:
                j = int(torch.randint(0, d, (1,), generator=generator, device=cpu).item())
                mask[i, j] = 0.0
    return mask.to(out_device)


def scale_hparams(n: int, d: int) -> dict:
    """SCALE: train-only size/d routing for architecture knobs."""
    if d >= 400:
        hidden, latent, depth = 384, 96, 3
        mask_rates = (0.1, 0.2, 0.35)
        score_k = 10
    elif d >= 64:
        hidden, latent, depth = 256, 64, 3
        mask_rates = (0.15, 0.3, 0.5)
        score_k = 16
    else:
        hidden, latent, depth = 128, 32, 2
        mask_rates = (0.2, 0.35, 0.5)
        score_k = 20

    if n < 200:
        depth = max(1, depth - 1)
        hidden = min(hidden, 128)
        score_k = min(score_k, 8)
    elif n > 10000:
        score_k = max(6, score_k - 2)

    return {
        "hidden": hidden,
        "latent": latent,
        "depth": depth,
        "mask_rates": mask_rates,
        "score_k": score_k,
        "dropout": 0.1 if n >= 500 else 0.05,
    }
