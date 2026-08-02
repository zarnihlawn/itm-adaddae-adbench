"""KALE — conflict-aware (anti-redundant) train-only fusion weights."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def fit_kale_weights(
    views: Dict[str, torch.Tensor],
    alpha0: float = 1.0,
    anchor: str = "reconstruction",
) -> Dict[str, float]:
    """
    Down-weight views that are highly redundant with the mean rank profile
    of other views (conflict / MI proxy via rank correlation).
    """
    keys = [k for k, v in views.items() if v is not None and v.numel() > 0]
    if not keys:
        return {"reconstruction": 1.0}
    if anchor not in keys:
        anchor = keys[0]

    mats = {}
    for k in keys:
        v = views[k].detach().float().cpu().numpy()
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        mats[k] = _rankdata(v)

    alphas = {}
    for k in keys:
        others = [mats[j] for j in keys if j != k]
        if not others:
            alphas[k] = float(alpha0)
            continue
        mean_other = np.mean(np.stack(others, axis=0), axis=0)
        if float(np.std(mats[k])) < 1e-12 or float(np.std(mean_other)) < 1e-12:
            corr = 0.0
        else:
            corr = float(np.corrcoef(mats[k], mean_other)[0, 1])
            if not np.isfinite(corr):
                corr = 0.0
        # High redundancy with coalition → lower weight
        conflict = max(0.0, corr)
        alphas[k] = float(alpha0 + max(0.0, 1.0 - conflict))

    # Also boost anchor agreement mildly
    r_anchor = mats[anchor]
    for k in keys:
        if float(np.std(mats[k])) < 1e-12:
            continue
        corr_a = float(np.corrcoef(r_anchor, mats[k])[0, 1])
        if np.isfinite(corr_a):
            alphas[k] += 0.5 * max(0.0, corr_a)

    s = sum(alphas.values()) + 1e-12
    return {k: float(v / s) for k, v in alphas.items()}


def kale_fuse(
    views: Dict[str, torch.Tensor],
    weights: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    w = weights or fit_kale_weights(views)

    def _norm(t: torch.Tensor) -> torch.Tensor:
        t = torch.nan_to_num(t.float(), nan=0.0, posinf=0.0, neginf=0.0)
        return t / t.mean().clamp_min(1e-12)

    scores = None
    for name, tensor in views.items():
        wi = float(w.get(name, 0.05))
        if not np.isfinite(wi):
            wi = 0.05
        term = wi * _norm(tensor)
        scores = term if scores is None else scores + term
    assert scores is not None
    return torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
