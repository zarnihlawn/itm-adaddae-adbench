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


def soften_mask_rates(rates: Sequence[float], factor: float = 0.7) -> tuple[float, ...]:
    """Lower mask rates for all-normal (semi) training — sharper normal recon."""
    return tuple(float(max(0.05, min(0.55, r * factor))) for r in rates)


def scale_hparams(n: int, d: int, *, semi: bool = False) -> dict:
    """SCALE: train-only size/d routing for architecture knobs.

    Phase 4: stronger high-d (CV/NLP) path; semi gets softer rates + more score_k.
    """
    if d >= 400:
        # Embedding / high-d: capacity + many light-mask score passes (G2/G3 killers)
        hidden, latent, depth = 512, 128, 3
        mask_rates = (0.08, 0.15, 0.28)
        score_k = 24
        dropout = 0.05
        high_mask_delta = 0.12
        high_mask_cap = 0.45
    elif d >= 64:
        hidden, latent, depth = 320, 80, 3
        mask_rates = (0.12, 0.25, 0.4)
        score_k = 18
        dropout = 0.08 if n >= 500 else 0.05
        high_mask_delta = 0.2
        high_mask_cap = 0.7
    else:
        hidden, latent, depth = 160, 40, 2
        mask_rates = (0.15, 0.3, 0.45)
        score_k = 20
        dropout = 0.08 if n >= 500 else 0.05
        high_mask_delta = 0.2
        high_mask_cap = 0.75

    if n < 200:
        # Tiny tables (glass): keep depth modest but do not starve score_k
        depth = max(1, depth - 1)
        hidden = min(hidden, 160)
        score_k = max(12, min(score_k, 16))
    elif n > 10000:
        # Huge tables (cover/fraud): still enough MCS passes
        score_k = max(12, score_k - 2)

    if semi:
        mask_rates = soften_mask_rates(mask_rates, factor=0.75)
        score_k = int(min(32, round(score_k * 1.25)))
        high_mask_delta = float(min(high_mask_delta, 0.15))
        high_mask_cap = float(min(high_mask_cap, 0.55))

    return {
        "hidden": hidden,
        "latent": latent,
        "depth": depth,
        "mask_rates": mask_rates,
        "score_k": score_k,
        "dropout": dropout,
        "high_mask_delta": high_mask_delta,
        "high_mask_cap": high_mask_cap,
        "semi": bool(semi),
    }
