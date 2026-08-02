"""PURA — Positive-Unlabeled Risk Alignment (nnPU-inspired) for contaminated train."""
from __future__ import annotations

from typing import Optional

import torch


def pura_sample_weights(
    per_sample_loss: torch.Tensor,
    prior_pi: float,
    min_weight: float = 0.05,
) -> torch.Tensor:
    """
    Soft nnPU-style weights: high-loss points treated as unlabeled (down-weighted).
    prior_pi ≈ contamination estimate ĉ from LF-DANC.
    """
    pi = float(np_clip(prior_pi, 0.01, 0.45))
    # Threshold at (1-π) quantile → mass π considered unlabeled
    q = torch.quantile(per_sample_loss.detach(), 1.0 - pi)
    # Soft logistic around q
    scale = (per_sample_loss.detach().std() + 1e-8) * 0.5
    soft = torch.sigmoid(-(per_sample_loss.detach() - q) / scale)
    w = soft * (1.0 - pi) + float(min_weight) * pi
    return w.clamp(min=float(min_weight), max=1.0)


def np_clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def pura_risk_adjust(
    loss_ps: torch.Tensor,
    weights: Optional[torch.Tensor],
) -> torch.Tensor:
    if weights is None:
        return loss_ps.mean()
    return (loss_ps * weights).sum() / weights.sum().clamp(min=1e-8)
