"""MAHALA — Ledoit–Wolf shrinkage Mahalanobis on [residual; latent]."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class MahalaState:
    mu: np.ndarray
    precision: np.ndarray  # Σ^{-1} in train-standardized space
    train_mean: np.ndarray
    train_std: np.ndarray


def _ledoit_wolf_precision(V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.covariance import LedoitWolf

    n, d = V.shape
    if n < 3 or d < 1:
        mu = V.mean(axis=0) if n else np.zeros(d, dtype=np.float64)
        return mu.astype(np.float64), np.eye(max(d, 1), dtype=np.float64)
    lw = LedoitWolf().fit(V)
    mu = lw.location_.astype(np.float64)
    cov = lw.covariance_.astype(np.float64)
    try:
        prec = np.linalg.pinv(cov, rcond=1e-5)
    except Exception:
        prec = np.eye(d, dtype=np.float64)
    return mu, prec


def fit_mahala(
    residuals: np.ndarray,
    latents: np.ndarray,
    max_samples: int = 4096,
) -> Optional[MahalaState]:
    R = np.asarray(residuals, dtype=np.float64)
    Z = np.asarray(latents, dtype=np.float64)
    if R.ndim != 2 or Z.ndim != 2 or R.shape[0] != Z.shape[0] or R.shape[0] < 4:
        return None
    V = np.concatenate([R, Z], axis=1)
    n = V.shape[0]
    if n > max_samples:
        rng = np.random.RandomState(0)
        V = V[rng.choice(n, max_samples, replace=False)]
    train_mean = V.mean(axis=0)
    train_std = V.std(axis=0) + 1e-8
    Vn = (V - train_mean) / train_std
    mu, prec = _ledoit_wolf_precision(Vn)
    return MahalaState(mu=mu, precision=prec, train_mean=train_mean, train_std=train_std)


@torch.inference_mode()
def mahala_score(
    residuals: torch.Tensor,
    latents: torch.Tensor,
    state: Optional[MahalaState],
) -> torch.Tensor:
    b = residuals.size(0)
    if state is None:
        return torch.zeros(b, device=residuals.device, dtype=torch.float32)
    V = torch.cat([residuals.float(), latents.float()], dim=1).cpu().numpy()
    Vn = (V - state.train_mean.reshape(1, -1)) / state.train_std.reshape(1, -1)
    delta = Vn - state.mu.reshape(1, -1)
    tmp = delta @ state.precision
    s = np.einsum("ij,ij->i", tmp, delta)
    return torch.tensor(s, device=residuals.device, dtype=torch.float32)
