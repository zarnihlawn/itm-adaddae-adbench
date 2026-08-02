"""SPARSE — Jaccard / presence residual view for sparse tabular."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def fit_sparse_prototype(X: np.ndarray, thresh: float = 1e-8) -> np.ndarray:
    """Mean presence pattern on train (binary-ish)."""
    B = (np.abs(X) > thresh).astype(np.float32)
    return B.mean(axis=0)


@torch.inference_mode()
def sparse_score(x: torch.Tensor, proto: Optional[np.ndarray], thresh: float = 1e-8) -> torch.Tensor:
    """1 - Jaccard(presence(x), proto soft)."""
    if proto is None:
        return torch.zeros(x.size(0), device=x.device, dtype=torch.float32)
    xb = (x.abs() > thresh).float()
    p = torch.tensor(proto, device=x.device, dtype=torch.float32).clamp(0, 1)
    # soft jaccard vs expected presence
    inter = (xb * p).sum(dim=1)
    union = (xb + p - xb * p).sum(dim=1).clamp_min(1e-8)
    j = inter / union
    s = 1.0 - j
    med = s.median().clamp_min(1e-8)
    return (s / med).float()
