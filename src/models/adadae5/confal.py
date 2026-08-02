"""CONFAL — train-only conformal / isotonic-like score calibration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class ConfalState:
    quantiles: np.ndarray  # empirical CDF knots on raw scores
    values: np.ndarray  # mapped [0,1] ranks


def fit_confal(
    scores: np.ndarray,
    y_normal_mask: Optional[np.ndarray] = None,
) -> Optional[ConfalState]:
    """
    Fit calibration on carved-val normals (semi) or all train scores (unsup null).
    Maps raw score → empirical CDF rank (higher = more anomalous).
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    if y_normal_mask is not None:
        m = np.asarray(y_normal_mask, dtype=bool).ravel()
        if m.shape[0] == s.shape[0] and m.any():
            # Use normals to define null; keep full score range for mapping
            ref = s[m]
        else:
            ref = s
    else:
        ref = s
    if ref.size < 8:
        return None
    ref = np.sort(ref)
    # Decimate knots
    k = min(256, ref.size)
    idx = np.linspace(0, ref.size - 1, k).astype(int)
    knots = ref[idx]
    vals = np.linspace(0.0, 1.0, k)
    return ConfalState(quantiles=knots, values=vals)


def confal_apply(scores: torch.Tensor, state: Optional[ConfalState]) -> torch.Tensor:
    if state is None:
        return scores
    s = scores.detach().float().cpu().numpy()
    mapped = np.interp(s, state.quantiles, state.values, left=0.0, right=1.0)
    return torch.tensor(mapped, device=scores.device, dtype=torch.float32)
