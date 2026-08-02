"""EPOCHÉ — val-loss plateau → shrink LR / expand SCS budget (train/val only)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class EpocheState:
    best_val: float = float("inf")
    stale: int = 0
    lr_shrinks: int = 0
    score_boost: int = 0
    history: List[float] = field(default_factory=list)


def epoche_step(
    state: EpocheState,
    val_loss: float,
    patience: int = 5,
    lr_factor: float = 0.5,
    max_shrinks: int = 3,
    scs_boost_step: int = 4,
) -> tuple[bool, float, int]:
    """
    Returns (should_shrink_lr, new_lr_factor_or_1, score_boost).
    """
    state.history.append(float(val_loss))
    improved = val_loss < state.best_val - 1e-6
    if improved:
        state.best_val = val_loss
        state.stale = 0
        return False, 1.0, state.score_boost
    state.stale += 1
    if state.stale < patience:
        return False, 1.0, state.score_boost
    state.stale = 0
    did = False
    factor = 1.0
    if state.lr_shrinks < max_shrinks:
        state.lr_shrinks += 1
        factor = lr_factor
        did = True
    state.score_boost += scs_boost_step
    return did, factor, state.score_boost
