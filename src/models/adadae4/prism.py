"""PRISM — whitened subspace residual scoring for high-d / CV / NLP."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA


class PrismState:
    def __init__(self, mean: np.ndarray, components: np.ndarray, whitener: np.ndarray):
        self.mean = mean
        self.components = components  # (k, d)
        self.whitener = whitener  # (k,)


def fit_prism(X: np.ndarray, max_components: int = 64, max_samples: int = 8000) -> Optional[PrismState]:
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    if d < 32 or n < 16:
        return None
    rng = np.random.RandomState(0)
    if n > max_samples:
        X = X[rng.choice(n, max_samples, replace=False)]
    k = min(max_components, X.shape[0] - 1, d)
    pca = PCA(n_components=k, svd_solver="randomized", random_state=0)
    Z = pca.fit_transform(X)
    # whitening scales
    var = Z.var(axis=0) + 1e-8
    return PrismState(
        mean=pca.mean_.astype(np.float32),
        components=pca.components_.astype(np.float32),
        whitener=(1.0 / np.sqrt(var)).astype(np.float32),
    )


@torch.inference_mode()
def prism_score(x: torch.Tensor, state: Optional[PrismState]) -> torch.Tensor:
    """||whitened(x) - proj|| residual magnitude (off-subspace + whitened energy)."""
    if state is None:
        return torch.zeros(x.size(0), device=x.device, dtype=torch.float32)
    xb = x.detach().cpu().numpy().astype(np.float64)
    centered = xb - state.mean
    # project
    z = centered @ state.components.T  # (n, k)
    zw = z * state.whitener
    recon = z @ state.components + state.mean
    off = np.linalg.norm(xb - recon, axis=1)
    on = np.linalg.norm(zw, axis=1)
    s = off + 0.1 * on
    med = np.median(s) + 1e-8
    out = torch.tensor((s / med).astype(np.float32), device=x.device)
    return out


def prism_should_disable_orbis(d: int, category: str = "classical") -> bool:
    return d >= 128 or category in ("cv", "nlp")
