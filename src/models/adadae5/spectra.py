"""SPECTRA — spectral FTP: Laplacian eigenmaps projected back to original d.

Keeps input_dim unchanged vs fair DDAE (no capacity leak from appending coords).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class SpectraState:
    evecs: np.ndarray  # (d, k)
    mean: np.ndarray
    scale: np.ndarray
    proj: np.ndarray  # (k, d) maps spectral coords → original feature space mix
    mix: float = 0.25  # blend of spectral reconstruction into X


def fit_spectra(
    X_train: np.ndarray,
    n_components: int = 8,
    max_samples: int = 4000,
) -> Optional[SpectraState]:
    X = np.asarray(X_train, dtype=np.float64)
    n, d = X.shape
    if n < 8 or d < 4:
        return None
    rng = np.random.RandomState(0)
    if n > max_samples:
        X = X[rng.choice(n, max_samples, replace=False)]
    mean = X.mean(axis=0)
    scale = X.std(axis=0) + 1e-8
    Xn = (X - mean) / scale
    # Avoid corrcoef NaNs on near-constant columns
    col_std = Xn.std(axis=0)
    active = col_std > 1e-8
    if int(active.sum()) < 4:
        return None
    Xa = Xn[:, active]
    C_sub = np.corrcoef(Xa, rowvar=False)
    C_sub = np.nan_to_num(C_sub, nan=0.0, posinf=0.0, neginf=0.0)
    C = np.zeros((d, d), dtype=np.float64)
    idx_a = np.where(active)[0]
    for i, ii in enumerate(idx_a):
        C[ii, idx_a] = C_sub[i]
    A = np.abs(C)
    np.fill_diagonal(A, 0.0)
    deg = A.sum(axis=1)
    D_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-8))
    L = np.eye(d) - (D_inv_sqrt[:, None] * A * D_inv_sqrt[None, :])
    try:
        evals, evecs = np.linalg.eigh(L)
    except Exception:
        return None
    k = min(n_components, d - 1, max(2, d // 4))
    idx = np.argsort(evals)[1 : k + 1]
    U = evecs[:, idx].astype(np.float64)  # (d, k)
    # Least-squares proj: coords @ proj ≈ Xn  →  proj = pinv(coords) @ Xn
    coords = Xn @ U  # (n, k)
    try:
        proj = np.linalg.pinv(coords) @ Xn  # (k, d)
    except Exception:
        proj = U.T
    return SpectraState(
        evecs=U,
        mean=mean,
        scale=scale,
        proj=proj.astype(np.float64),
        mix=0.25,
    )


def spectra_transform(X: np.ndarray, state: Optional[SpectraState]) -> np.ndarray:
    """Blend spectral reconstruction into X; output shape == input shape."""
    X = np.asarray(X, dtype=np.float32)
    if state is None:
        return X
    Xn = (X.astype(np.float64) - state.mean) / state.scale
    coords = Xn @ state.evecs
    recon = coords @ state.proj
    mixed = (1.0 - state.mix) * Xn + state.mix * recon
    out = mixed * state.scale + state.mean
    return out.astype(np.float32)


def spectra_fit_transform(
    X_train: np.ndarray,
    n_components: int = 8,
) -> Tuple[np.ndarray, Optional[SpectraState]]:
    st = fit_spectra(X_train, n_components=n_components)
    return spectra_transform(X_train, st), st
