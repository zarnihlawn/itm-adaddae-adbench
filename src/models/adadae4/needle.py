"""NEEDLE — rare-contamination tail-focused fusion / conformal α."""
from __future__ import annotations

from typing import Dict

import torch


def needle_aegis_alpha(contam: float, base: float = 0.1) -> float:
    """Rare anomalies → tighter conformal band (smaller alpha = higher quantile)."""
    c = float(max(contam, 1e-4))
    # For rare: alpha ~ 0.02–0.05; for common: base
    if c <= 0.02:
        return float(max(0.02, min(0.05, c * 2)))
    return base


@torch.inference_mode()
def needle_tail_boost(scores: torch.Tensor, tail_frac: float = 0.05) -> torch.Tensor:
    """Amplify separation in the upper tail (rare AD)."""
    s = scores.float()
    q = torch.quantile(s, 1.0 - tail_frac)
    boost = torch.relu(s - q)
    return s + boost
