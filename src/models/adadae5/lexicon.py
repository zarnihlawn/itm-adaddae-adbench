"""LEXICON — Dirichlet fusion weights from train-only val recon rank consistency."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def fit_lexicon_weights(
    views: Dict[str, torch.Tensor],
    alpha0: float = 1.0,
    anchor: str = "reconstruction",
) -> Dict[str, float]:
    """
    Fit Dirichlet-like weights: views that rank-agree with anchor get higher mass.
    Never uses test labels — only train/val score tensors.
    """
    keys = [k for k, v in views.items() if v is not None and v.numel() > 0]
    if not keys:
        return {"reconstruction": 1.0}
    if anchor not in keys:
        anchor = keys[0]

    anchor_np = views[anchor].detach().float().cpu().numpy()
    anchor_np = np.nan_to_num(anchor_np, nan=0.0, posinf=0.0, neginf=0.0)
    r_anchor = _rankdata(anchor_np)
    alphas = {}
    for k in keys:
        v = views[k].detach().float().cpu().numpy()
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        if v.shape != anchor_np.shape:
            alphas[k] = float(alpha0)
            continue
        if float(np.std(v)) < 1e-12 or float(np.std(anchor_np)) < 1e-12:
            alphas[k] = float(alpha0)
            continue
        rk = _rankdata(v)
        corr = float(np.corrcoef(r_anchor, rk)[0, 1]) if len(r_anchor) > 2 else 0.0
        if not np.isfinite(corr):
            corr = 0.0
        alphas[k] = float(alpha0 + max(0.0, corr))

    s = sum(alphas.values()) + 1e-12
    return {k: float(v / s) for k, v in alphas.items()}


def lexicon_fuse(
    views: Dict[str, torch.Tensor],
    weights: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    if not views:
        raise ValueError("no views")
    w = weights or fit_lexicon_weights(views)

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
