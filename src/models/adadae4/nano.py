"""NANO — tiny-n capacity / neighbor shrink."""
from __future__ import annotations

from typing import Dict, List, Tuple


def nano_model_dims(n: int, hidden_dims: List[int], latent_dim: int) -> Tuple[List[int], int]:
    if n >= 300:
        return list(hidden_dims), int(latent_dim)
    # Cap width for tiny data
    capped = [min(h, 128 if n < 150 else 256) for h in hidden_dims]
    lat = min(int(latent_dim), 16 if n < 150 else 24)
    return capped, lat


def nano_neighbors(n: int, geode_k: int, plexus_k: int) -> Tuple[int, int]:
    if n >= 300:
        return geode_k, plexus_k
    g = max(4, min(geode_k, max(4, n // 8)))
    p = max(2, min(plexus_k, max(2, n // 10)))
    return g, p


def nano_disable_heavy(n: int) -> Dict[str, bool]:
    """Disable expensive / unstable views on tiny-n."""
    if n >= 300:
        return {}
    return {
        "use_flux": False,
        "use_mirage": False,
        "use_hydra": False,
    }


def nano_weight_decay(n: int, base: float = 0.0) -> float:
    if n < 150:
        return max(base, 1e-3)
    if n < 300:
        return max(base, 5e-4)
    return base
