"""SIEVE — contamination-aware rejection + optional IF train prior."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


def sieve_rejection_quantile(contam: float, base: float = 0.95) -> float:
    """Heavier contamination → lower keep-quantile (reject more)."""
    c = float(np.clip(contam, 0.01, 0.4))
    q = base - 0.5 * c  # e.g. c=0.3 → 0.80
    return float(np.clip(q, 0.7, 0.98))


def fit_sieve_iforest(X: np.ndarray, seed: int = 0, max_samples: int = 10000):
    from sklearn.ensemble import IsolationForest

    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    rng = np.random.RandomState(seed)
    if n > max_samples:
        X = X[rng.choice(n, max_samples, replace=False)]
    clf = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=seed,
        n_jobs=1,
    )
    clf.fit(X)
    return clf


@torch.inference_mode()
def sieve_iforest_view(X: torch.Tensor, clf) -> torch.Tensor:
    """Higher = more anomalous (negate decision_function)."""
    xb = X.detach().cpu().numpy().astype(np.float32)
    s = -clf.decision_function(xb)
    s = s - s.min()
    med = np.median(s) + 1e-8
    return torch.tensor((s / med).astype(np.float32), device=X.device)
