"""CALIX: Conformal Adaptive Likelihood Index for multi-view fusion.

Train-normal quantile calibration → adaptive view weights (no test labels).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch


def _norm(v: torch.Tensor) -> torch.Tensor:
    return v / v.mean().clamp_min(1e-12)


@torch.inference_mode()
def fit_calix_weights(
    view_scores: Dict[str, torch.Tensor],
    base_weights: Optional[Dict[str, float]] = None,
    quantile: float = 0.9,
    epsilon: float = 1e-6,
) -> Dict[str, float]:
    """Estimate fusion weights from train-normal view score tails.

    Views with tighter normal tails (lower high-quantile) get higher weight
    (more reliable normality signal). Weights are renormalized to sum to 1
    over active views.
    """
    base = base_weights or {}
    reliab: Dict[str, float] = {}
    for name, scores in view_scores.items():
        if scores is None or scores.numel() == 0:
            continue
        s = scores.float().reshape(-1)
        q = torch.quantile(s, quantile).item()
        iqr = (torch.quantile(s, 0.75) - torch.quantile(s, 0.25)).item()
        # High reliability = low dispersion of normals
        r = 1.0 / (abs(q) + abs(iqr) + epsilon)
        prior = float(base.get(name, 1.0))
        reliab[name] = r * max(prior, 1e-6)
    total = sum(reliab.values()) + epsilon
    return {k: v / total for k, v in reliab.items()}


@torch.inference_mode()
def calix_conformal_pvalues(
    test_scores: torch.Tensor,
    train_normal_scores: torch.Tensor,
) -> torch.Tensor:
    """Conformal p-values: fraction of train normals with score >= test score."""
    # Higher anomaly score => lower p
    tn = train_normal_scores.float().reshape(-1)
    ts = test_scores.float().reshape(-1)
    # (n_test, n_train)
    ge = (tn.unsqueeze(0) >= ts.unsqueeze(1)).float()
    return (1.0 + ge.sum(dim=1)) / (1.0 + float(tn.numel()))


def fuse_calix_views(
    views: Dict[str, torch.Tensor],
    weights: Dict[str, float],
) -> torch.Tensor:
    """Weighted sum of mean-normalized active views."""
    parts = []
    for name, tensor in views.items():
        w = float(weights.get(name, 0.0))
        if w <= 0.0 or tensor is None:
            continue
        parts.append(w * _norm(tensor.float()))
    if not parts:
        # Fallback: first view
        any_v = next(iter(views.values()))
        return _norm(any_v.float())
    return sum(parts)
