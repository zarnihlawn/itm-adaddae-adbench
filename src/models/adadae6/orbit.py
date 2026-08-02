"""ORBIT — embedding cosine / whitened residual anomaly view."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class OrbitState:
    mean: np.ndarray
    scale: np.ndarray


def fit_orbit(X_train: np.ndarray) -> Optional[OrbitState]:
    X = np.asarray(X_train, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 4:
        return None
    mean = X.mean(axis=0)
    scale = X.std(axis=0) + 1e-8
    return OrbitState(mean=mean, scale=scale)


@torch.inference_mode()
def orbit_score(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    state: Optional[OrbitState] = None,
) -> torch.Tensor:
    """1 - cosine(x, x̂) plus mild whitened L2; works for tabular and embeds."""
    xf = x.float()
    hf = x_hat.float()
    if state is not None:
        mean = torch.tensor(state.mean, device=xf.device, dtype=xf.dtype)
        scale = torch.tensor(state.scale, device=xf.device, dtype=xf.dtype)
        xf = (xf - mean) / scale
        hf = (hf - mean) / scale
    xn = torch.nn.functional.normalize(xf, dim=1, eps=1e-8)
    hn = torch.nn.functional.normalize(hf, dim=1, eps=1e-8)
    cos = (xn * hn).sum(dim=1).clamp(-1.0, 1.0)
    cos_score = 1.0 - cos
    l2 = torch.norm(xf - hf, dim=1)
    return cos_score + 0.1 * l2
