"""GEODE: Geometry-aware Off-manifold Deviation Estimate.

Local PCA / tangent residual in latent space on a train memory bank.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch


@torch.inference_mode()
def build_geode_basis(
    z_memory: torch.Tensor,
    n_neighbors: int = 32,
    n_components: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precompute global PCA basis on memory (fast tabular proxy for local PCA).

    Returns (mu, components, explained) where components is (k, d).
    For true local PCA at query time we still use nearest neighbors + this fallback.
    """
    if z_memory.numel() == 0:
        d = 1
        return (
            torch.zeros(d),
            torch.eye(1),
            torch.ones(1),
        )
    z = z_memory.float()
    mu = z.mean(dim=0)
    xc = z - mu
    # Economy SVD
    k = max(1, min(n_components, z.size(0) - 1, z.size(1)))
    try:
        _, s, vh = torch.linalg.svd(xc, full_matrices=False)
        comps = vh[:k]
        ev = (s[:k] ** 2)
        ev = ev / ev.sum().clamp_min(1e-12)
    except RuntimeError:
        comps = torch.eye(z.size(1), device=z.device)[:k]
        ev = torch.ones(k, device=z.device) / k
    return mu, comps, ev


@torch.inference_mode()
def geode_score(
    z_query: torch.Tensor,
    z_memory: torch.Tensor,
    n_neighbors: int = 32,
    n_components: int = 4,
    global_mu: Optional[torch.Tensor] = None,
    global_comps: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Off-manifold residual: distance to local (or global) tangent subspace.

    s = || (I - P_k)(z - mu_k) ||_2
    """
    if z_memory.size(0) == 0:
        return torch.zeros(z_query.size(0), device=z_query.device, dtype=torch.float32)

    b = z_query.size(0)
    scores = torch.zeros(b, device=z_query.device, dtype=torch.float32)
    k_nn = min(n_neighbors, z_memory.size(0))
    k_pca = max(1, min(n_components, k_nn - 1, z_query.size(1)))

    # Batch cdist can be heavy; chunk queries
    chunk = 256
    for i in range(0, b, chunk):
        q = z_query[i : i + chunk]
        dist = torch.cdist(q, z_memory, p=2)
        knn_d, knn_idx = torch.topk(dist, k=k_nn, dim=1, largest=False)
        for j in range(q.size(0)):
            neigh = z_memory[knn_idx[j]]
            mu = neigh.mean(dim=0)
            xc = neigh - mu
            if k_nn < 3 or q.size(1) < 2:
                # Fallback: knn distance
                scores[i + j] = knn_d[j].mean()
                continue
            try:
                # Local PCA via SVD of neighborhood
                _, _, vh = torch.linalg.svd(xc, full_matrices=False)
                comps = vh[:k_pca]
                delta = q[j] - mu
                proj = (delta @ comps.T) @ comps
                resid = delta - proj
                scores[i + j] = torch.norm(resid)
            except RuntimeError:
                if global_mu is not None and global_comps is not None:
                    delta = q[j] - global_mu
                    proj = (delta @ global_comps.T) @ global_comps
                    scores[i + j] = torch.norm(delta - proj)
                else:
                    scores[i + j] = knn_d[j].mean()
    # Normalize by median for scale stability
    med = scores.median().clamp_min(1e-8)
    return (scores / med).float()
