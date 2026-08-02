"""POLIS — multi-prototype memory for multimodal normals."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans


def fit_polis_prototypes(
    Z: np.ndarray,
    cluster_sep: float = 0.0,
    max_k: int = 5,
    max_samples: int = 8000,
) -> Optional[np.ndarray]:
    """Return (k, d) prototypes in latent/feature space."""
    Z = np.asarray(Z, dtype=np.float64)
    n, d = Z.shape
    if n < 16:
        return None
    k = 2
    if cluster_sep > 2.0:
        k = 3
    if cluster_sep > 3.0:
        k = min(max_k, 4)
    k = min(k, max(2, n // 20))
    rng = np.random.RandomState(0)
    if n > max_samples:
        Z = Z[rng.choice(n, max_samples, replace=False)]
    km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3, batch_size=256)
    km.fit(Z)
    return km.cluster_centers_.astype(np.float32)


@torch.inference_mode()
def polis_score(z_query: torch.Tensor, prototypes: Optional[np.ndarray]) -> torch.Tensor:
    """Distance to nearest prototype (normalized)."""
    if prototypes is None or len(prototypes) == 0:
        return torch.zeros(z_query.size(0), device=z_query.device, dtype=torch.float32)
    proto = torch.tensor(prototypes, device=z_query.device, dtype=torch.float32)
    dist = torch.cdist(z_query.float(), proto, p=2)
    s = dist.min(dim=1).values
    med = s.median().clamp_min(1e-8)
    return (s / med).float()
