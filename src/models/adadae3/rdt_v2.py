"""RDT-v2 — soft continuous rejection via logistic of loss rank + EMA."""
from __future__ import annotations

from typing import Optional, Tuple

import torch


class SoftRejectionEMA:
    def __init__(self, momentum: float = 0.9):
        self.momentum = momentum
        self.center: Optional[float] = None
        self.scale: Optional[float] = None

    def update(self, losses: torch.Tensor) -> None:
        med = float(losses.median().item())
        mad = float((losses - med).abs().median().item()) + 1e-8
        if self.center is None:
            self.center, self.scale = med, mad
        else:
            self.center = self.momentum * self.center + (1 - self.momentum) * med
            self.scale = self.momentum * self.scale + (1 - self.momentum) * mad

    def weights(
        self,
        losses: torch.Tensor,
        min_weight: float = 0.1,
        steepness: float = 4.0,
    ) -> torch.Tensor:
        if self.center is None:
            self.update(losses)
        assert self.center is not None and self.scale is not None
        z = (losses - self.center) / self.scale
        # High loss → low weight
        w = torch.sigmoid(-steepness * z)
        return w.clamp(min=min_weight, max=1.0)


def soft_rejection_weights(
    losses: torch.Tensor,
    state: SoftRejectionEMA,
    min_weight: float = 0.1,
    steepness: float = 4.0,
    update: bool = True,
) -> Tuple[torch.Tensor, SoftRejectionEMA]:
    if update:
        state.update(losses.detach())
    return state.weights(losses.detach(), min_weight=min_weight, steepness=steepness), state
