"""MIRAGE — CPU-safe epistemic variance via dropout/noise seeds (replaces VUS on CPU)."""
from __future__ import annotations

from typing import Callable, List

import torch


@torch.inference_mode()
def mirage_score(
    score_fn: Callable[[], torch.Tensor],
    n_draws: int = 5,
) -> torch.Tensor:
    """Run score_fn n_draws times; return std across draws (epistemic)."""
    draws: List[torch.Tensor] = []
    for _ in range(max(2, n_draws)):
        draws.append(score_fn().float())
    stack = torch.stack(draws, dim=0)
    return stack.std(dim=0).clamp_min(0.0)


@torch.inference_mode()
def mirage_from_residuals(residuals: torch.Tensor) -> torch.Tensor:
    """If residuals is (draws, N), return std; if (N,), return zeros."""
    if residuals.dim() == 1:
        return torch.zeros_like(residuals)
    return residuals.float().std(dim=0).clamp_min(0.0)
