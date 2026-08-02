"""KAIROS — dual schedule: train curriculum T vs score SSTS budget."""
from __future__ import annotations

from typing import List, Tuple


def kairos_train_T(epoch: int, epochs: int, T_full: int, T_min: int = 10) -> int:
    """Linear curriculum: grow effective max-t used in training."""
    if epochs <= 1:
        return T_full
    frac = min(1.0, max(0.0, (epoch + 1) / float(epochs)))
    t = int(round(T_min + frac * (T_full - T_min)))
    return max(T_min, min(T_full, t))


def kairos_score_budget(T: int, scs_max: int, boost: int = 0) -> int:
    """Scoring timestep budget (can expand via EPOCHÉ)."""
    return max(4, min(T - 1, scs_max + max(0, boost)))


def kairos_split(
    epoch: int,
    epochs: int,
    T: int,
    scs_max: int,
    score_boost: int = 0,
) -> Tuple[int, int]:
    return kairos_train_T(epoch, epochs, T), kairos_score_budget(T, scs_max, score_boost)
