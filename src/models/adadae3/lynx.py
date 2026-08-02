"""LYNX — rank-based multi-view aggregation (scale-robust)."""
from __future__ import annotations

from typing import Dict

import torch


@torch.inference_mode()
def fuse_lynx_views(views: Dict[str, torch.Tensor]) -> torch.Tensor:
    names = [k for k, v in views.items() if v is not None]
    if not names:
        raise ValueError("No views for LYNX")
    ranks = []
    for n in names:
        s = views[n].float()
        # Average rank (higher score → higher rank)
        order = torch.argsort(s)
        r = torch.empty_like(s)
        r[order] = torch.arange(s.numel(), device=s.device, dtype=torch.float32)
        ranks.append(r)
    return torch.stack(ranks, dim=0).mean(dim=0)
