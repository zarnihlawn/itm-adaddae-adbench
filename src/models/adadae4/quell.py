"""QUELL — soft view reliability floor / dropout at fusion."""
from __future__ import annotations

from typing import Dict, Optional

import torch


@torch.inference_mode()
def quell_reliability(
    views: Dict[str, torch.Tensor],
    base: Optional[Dict[str, float]] = None,
    floor: float = 0.05,
    conflict_penalty: float = 0.5,
) -> Dict[str, float]:
    """Down-weight conflicting / weak views; enforce reliability floor."""
    names = list(views.keys())
    if not names:
        return {}
    stacked = torch.stack([views[n].float() for n in names], dim=0)
    # rank-correlate each view with mean rank
    mean_s = stacked.mean(dim=0, keepdim=True)
    corr = []
    for i, n in enumerate(names):
        a = stacked[i] - stacked[i].mean()
        b = mean_s.squeeze(0) - mean_s.mean()
        num = (a * b).sum()
        den = a.norm() * b.norm() + 1e-8
        c = float((num / den).clamp(-1, 1).item())
        corr.append(max(0.0, c))
    rel = {}
    for n, c in zip(names, corr):
        b = float((base or {}).get(n, 1.0))
        r = b * (0.5 + 0.5 * c)
        # conflict: if view far from mean
        sn = stacked[names.index(n)]
        conflict = float((sn - mean_s.squeeze(0)).abs().mean().item())
        r = r / (1.0 + conflict_penalty * conflict)
        rel[n] = max(floor, r)
    s = sum(rel.values()) or 1.0
    return {k: v / s for k, v in rel.items()}


@torch.inference_mode()
def quell_fuse(views: Dict[str, torch.Tensor], rel: Dict[str, float]) -> torch.Tensor:
    acc = None
    for k, v in views.items():
        w = float(rel.get(k, 0.0))
        term = w * v.float()
        acc = term if acc is None else acc + term
    assert acc is not None
    return acc
