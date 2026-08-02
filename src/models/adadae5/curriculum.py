"""CURRICULUM-SNR — epoch-wise easy→hard noise curriculum."""
from __future__ import annotations


def curriculum_t_max(epoch: int, epochs: int, T: int, warmup_frac: float = 0.3) -> int:
    """t_max(e) monotone in epoch: start ~T/4, end at T."""
    e = max(0, int(epoch))
    E = max(1, int(epochs))
    warm = max(1, int(E * float(warmup_frac)))
    if e < warm:
        frac = 0.25 + 0.75 * (e / warm)
    else:
        frac = 1.0
    return max(2, int(round(T * frac)))
