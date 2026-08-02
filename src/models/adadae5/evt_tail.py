"""EVT-TAIL — peaks-over-threshold GPD transform of anomaly scores."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class EvtState:
    u: float
    xi: float
    beta: float
    exceed_rate: float


def fit_evt_gpd(
    scores: np.ndarray,
    quantile: float = 0.90,
    max_samples: int = 10000,
) -> Optional[EvtState]:
    s = np.asarray(scores, dtype=np.float64).ravel()
    if s.size < 32:
        return None
    if s.size > max_samples:
        rng = np.random.RandomState(0)
        s = s[rng.choice(s.size, max_samples, replace=False)]
    u = float(np.quantile(s, quantile))
    exc = s[s > u] - u
    if exc.size < 8:
        return None
    # Method of moments for GPD
    mean = float(exc.mean())
    var = float(exc.var()) + 1e-12
    xi = 0.5 * (1.0 - (mean**2) / var)
    xi = float(np.clip(xi, -0.4, 0.8))
    beta = float(mean * (1.0 - xi))
    beta = max(beta, 1e-8)
    rate = float((s > u).mean())
    return EvtState(u=u, xi=xi, beta=beta, exceed_rate=max(rate, 1e-6))


def evt_tail_transform(scores: torch.Tensor, state: Optional[EvtState]) -> torch.Tensor:
    """Map raw scores to extreme severity; identity if unfit."""
    if state is None:
        return torch.nan_to_num(scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
    s = torch.nan_to_num(scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
    u, xi, beta, rate = state.u, state.xi, state.beta, state.exceed_rate
    over = torch.clamp(s - u, min=0.0)
    if abs(xi) < 1e-6:
        sf = torch.exp(-over / beta)
    else:
        sf = torch.pow(1.0 + xi * over / beta, -1.0 / xi).clamp(min=1e-12, max=1.0)
    sev = -torch.log(sf.clamp(min=1e-12)) + (-np.log(max(rate, 1e-6)))
    below = s <= u
    smin = s.min()
    smax = s.max()
    mild = (s - smin) / (smax - smin + 1e-8)
    out = torch.where(below, mild * float(sev.median().clamp(min=1e-3)), sev)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
