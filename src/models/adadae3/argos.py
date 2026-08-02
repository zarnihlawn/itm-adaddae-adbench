"""ARGOS — Dirichlet / evidential multi-view fusion."""
from __future__ import annotations

from typing import Dict

import torch


@torch.inference_mode()
def fuse_argos_views(
    views: Dict[str, torch.Tensor],
    reliability: Dict[str, float] | None = None,
) -> torch.Tensor:
    """
    Convert each view to evidence e_i = rel_i / (1 + score_i), then Dirichlet
    strength S = sum e_i + K; fused score = conflict-aware weighted mean.
    """
    names = [k for k, v in views.items() if v is not None]
    if not names:
        raise ValueError("No views for ARGOS")
    rel = reliability or {n: 1.0 for n in names}
    stacked = torch.stack([views[n].float() for n in names], dim=0)  # (V, N)
    # Normalize each view
    med = stacked.median(dim=1, keepdim=True).values.clamp_min(1e-8)
    sn = stacked / med
    e = []
    for i, n in enumerate(names):
        r = float(rel.get(n, 1.0))
        e.append(r / (1.0 + sn[i]))
    evidence = torch.stack(e, dim=0)  # (V, N)
    S = evidence.sum(dim=0) + len(names)
    belief = evidence / S.unsqueeze(0)
    # Conflict: high disagreement → down-weight
    mean_s = sn.mean(dim=0, keepdim=True)
    conflict = (sn - mean_s).abs().mean(dim=0)
    conf_w = 1.0 / (1.0 + conflict)
    fused = (belief * sn).sum(dim=0) * conf_w
    return fused.float()
