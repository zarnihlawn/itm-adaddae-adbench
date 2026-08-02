"""LOCUS — latent LOF / density-ratio anomaly view on train memory."""
from __future__ import annotations

from typing import Optional

import torch


@torch.inference_mode()
def locus_lof_score(
    z: torch.Tensor,
    memory: torch.Tensor,
    k: int = 5,
) -> torch.Tensor:
    """
    Simplified LOF: ratio of mean kNN distance in query vs local density on memory.
    Higher = more anomalous.
    """
    if memory is None or memory.numel() == 0 or z.numel() == 0:
        return torch.zeros(z.size(0), device=z.device, dtype=torch.float32)
    mem = memory.float().to(z.device)
    zf = z.float()
    n_mem = mem.size(0)
    kk = max(1, min(int(k), n_mem - 1 if n_mem > 1 else 1))
    # distances query→memory
    # (B, M)
    d = torch.cdist(zf, mem)
    knn_q, _ = torch.topk(d, k=kk, largest=False, dim=1)
    reach_q = knn_q.mean(dim=1).clamp_min(1e-8)

    # local density on memory: each mem point's mean kNN among memory
    if n_mem > kk + 1:
        d_mm = torch.cdist(mem, mem)
        # exclude self: set diag large
        d_mm.fill_diagonal_(float("inf"))
        knn_m, _ = torch.topk(d_mm, k=kk, largest=False, dim=1)
        reach_m = knn_m.mean(dim=1).clamp_min(1e-8)
        # for each query, average reach of its neighbors
        _, idx = torch.topk(d, k=kk, largest=False, dim=1)
        neigh_reach = reach_m[idx].mean(dim=1)
    else:
        neigh_reach = reach_q
    lof = reach_q / neigh_reach.clamp_min(1e-8)
    return torch.nan_to_num(lof.float(), nan=1.0, posinf=10.0, neginf=1.0)
