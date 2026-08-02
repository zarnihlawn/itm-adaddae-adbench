"""NAUTILUS — tiny-n capacity / neighbor / memory shrink."""
from __future__ import annotations

from typing import List, Tuple


def nautilus_should_apply(n: int, n_star: int = 300) -> bool:
    return int(n) > 0 and int(n) < int(n_star)


def nautilus_model_dims(
    n: int,
    hidden_dims: List[int],
    latent_dim: int,
    n_star: int = 300,
) -> Tuple[List[int], int]:
    """Shrink hidden/latent when n is tiny."""
    if not nautilus_should_apply(n, n_star):
        return list(hidden_dims), int(latent_dim)
    scale = max(0.25, min(1.0, (float(n) / float(n_star)) ** 0.5))
    h = [max(32, int(round(x * scale))) for x in hidden_dims]
    lat = max(8, int(round(latent_dim * scale)))
    return h, lat


def nautilus_neighbors(n: int, geode_k: int, dte_k: int, n_star: int = 300) -> Tuple[int, int]:
    if not nautilus_should_apply(n, n_star):
        return int(geode_k), int(dte_k)
    scale = max(0.35, min(1.0, (float(n) / float(n_star)) ** 0.5))
    return max(4, int(round(geode_k * scale))), max(3, int(round(dte_k * scale)))


def nautilus_memory_size(n: int, base: int, n_star: int = 300) -> int:
    if not nautilus_should_apply(n, n_star):
        return int(base)
    return max(64, min(int(base), max(64, int(n) * 4)))
