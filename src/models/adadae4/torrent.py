"""TORRENT — huge-n reservoir memory + adaptive scoring budgets."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def torrent_memory_size(n: int, base: int = 4096) -> int:
    if n < 50_000:
        return base
    # Cap memory bank for huge-n
    return min(base, max(512, int(2048)))


def torrent_score_batch(n: int, base: int = 1024) -> int:
    if n < 50_000:
        return base
    return min(base, 256)


def torrent_ssl_subsample(x: torch.Tensor, max_n: int = 4096) -> torch.Tensor:
    if x.size(0) <= max_n:
        return x
    idx = torch.randperm(x.size(0), device=x.device)[:max_n]
    return x[idx]


def torrent_reservoir_indices(n: int, capacity: int, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    if n <= capacity:
        return np.arange(n)
    return rng.choice(n, capacity, replace=False)
