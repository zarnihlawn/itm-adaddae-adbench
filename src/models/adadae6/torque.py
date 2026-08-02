"""TORQUE — huge-n memory / score-batch / train-subsample caps."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def torque_should_apply(n: int, n_star: int = 50_000) -> bool:
    return int(n) >= int(n_star)


def torque_memory_size(n: int, base: int, n_star: int = 50_000) -> int:
    if not torque_should_apply(n, n_star):
        return int(base)
    # Cap bank growth
    return min(int(base), 2048)


def torque_score_batch(n: int, base: int, n_star: int = 50_000) -> int:
    if not torque_should_apply(n, n_star):
        return int(base)
    return min(int(base), 512)


def torque_train_subsample(
    X: np.ndarray,
    max_n: int = 40_000,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X_sub, index_into_original). Identity if n small."""
    X = np.asarray(X)
    n = X.shape[0]
    if n <= max_n:
        return X, np.arange(n)
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, max_n, replace=False)
    idx.sort()
    return X[idx], idx
