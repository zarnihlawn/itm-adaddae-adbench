"""ROBUST — MAD-normalize views before fusion."""
from __future__ import annotations

from typing import Dict

import torch


@torch.inference_mode()
def mad_normalize(v: torch.Tensor) -> torch.Tensor:
    med = v.median()
    mad = (v - med).abs().median().clamp_min(1e-8)
    return (v - med) / (1.4826 * mad)


@torch.inference_mode()
def robust_normalize_views(views: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: mad_normalize(v.float()) for k, v in views.items() if v is not None}
