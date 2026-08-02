"""PLEXUS — graph-kNN message residual on latent memory bank."""
from __future__ import annotations

import torch


@torch.inference_mode()
def plexus_score(
    z_query: torch.Tensor,
    z_memory: torch.Tensor,
    n_neighbors: int = 16,
    chunk: int = 256,
) -> torch.Tensor:
    """|| z - mean(kNN(z)) || after one message-pass average."""
    if z_memory.numel() == 0:
        return torch.zeros(z_query.size(0), device=z_query.device)
    k = min(n_neighbors, z_memory.size(0))
    out = torch.zeros(z_query.size(0), device=z_query.device, dtype=torch.float32)
    for i in range(0, z_query.size(0), chunk):
        q = z_query[i : i + chunk]
        dist = torch.cdist(q.float(), z_memory.float(), p=2)
        _, idx = torch.topk(dist, k=k, dim=1, largest=False)
        neigh = z_memory[idx]  # (b, k, d)
        agg = neigh.mean(dim=1)
        out[i : i + q.size(0)] = torch.norm(q - agg, dim=1)
    med = out.median().clamp_min(1e-8)
    return (out / med).float()
