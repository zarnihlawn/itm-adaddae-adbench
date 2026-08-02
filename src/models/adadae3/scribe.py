"""SCRIBE — SAM-lite sharpness-aware step every k epochs."""
from __future__ import annotations

from typing import Iterable

import torch


def scribe_should_run(epoch: int, every_k: int) -> bool:
    return every_k > 0 and (epoch + 1) % every_k == 0


@torch.no_grad()
def scribe_ascent(params: Iterable[torch.nn.Parameter], rho: float = 0.05) -> list:
    """Perturb params along grad; return list of (param, e_w) to restore."""
    backups = []
    for p in params:
        if p.grad is None:
            continue
        norm = p.grad.norm()
        if norm < 1e-12:
            continue
        e_w = (rho / norm) * p.grad
        p.add_(e_w)
        backups.append((p, e_w))
    return backups


@torch.no_grad()
def scribe_restore(backups: list) -> None:
    for p, e_w in backups:
        p.sub_(e_w)
