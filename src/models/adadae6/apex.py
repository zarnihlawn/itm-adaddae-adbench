"""APEX — contam-aware rare-tail / log-odds map on fused scores."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class ApexState:
    median: float
    mad: float
    alpha: float  # contam-aware sharpness
    q_hi: float


def fit_apex(
    scores: np.ndarray,
    contamination: float = 0.05,
) -> Optional[ApexState]:
    s = np.asarray(scores, dtype=np.float64).ravel()
    s = s[np.isfinite(s)]
    if s.size < 16:
        return None
    med = float(np.median(s))
    mad = float(np.median(np.abs(s - med))) + 1e-8
    c = float(np.clip(contamination, 0.005, 0.35))
    # Sharper map when rare (low c)
    alpha = float(np.clip(1.5 / max(c, 0.01), 2.0, 40.0))
    q_hi = float(np.quantile(s, 1.0 - c))
    return ApexState(median=med, mad=mad, alpha=alpha, q_hi=q_hi)


def apex_transform(scores: torch.Tensor, state: Optional[ApexState]) -> torch.Tensor:
    """Monotone log-odds-like severity; identity if unfit."""
    if state is None:
        return torch.nan_to_num(scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
    s = torch.nan_to_num(scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
    z = (s - state.median) / state.mad
    # Softplus-style rare boost above high quantile
    boost = torch.nn.functional.softplus(state.alpha * (s - state.q_hi) / (state.mad + 1e-8))
    out = z + boost
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
