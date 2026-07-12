"""Diffusion Time Estimation (DTE) scoring for AdaDDAE.

Inspired by: On Diffusion Modeling for Anomaly Detection (ICLR 2024).
Anomalies are off-manifold → higher posterior mass on large diffusion timesteps.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def build_latent_memory(
    z_train: torch.Tensor,
    max_samples: int = 4096,
    seed: int = 0,
) -> torch.Tensor:
    """Subsample training latents for kNN DTE (memory bank)."""
    n = z_train.size(0)
    if n <= max_samples:
        return z_train.detach()
    g = torch.Generator(device=z_train.device)
    g.manual_seed(seed)
    idx = torch.randperm(n, generator=g, device=z_train.device)[:max_samples]
    return z_train[idx].detach()


def knn_dte_score(
    z_query: torch.Tensor,
    z_memory: torch.Tensor,
    k: int = 5,
    T: int = 50,
) -> torch.Tensor:
    """
    Non-parametric DTE proxy: expected diffusion time scales with kNN distance.
    s_DTE(x) ∝ d_kNN(z(x)) — farther from train manifold → larger effective t.
    """
    if z_memory.size(0) == 0:
        return torch.zeros(z_query.size(0), device=z_query.device, dtype=torch.float32)
    k = min(k, z_memory.size(0))
    # (B, M) squared distances
    dist = torch.cdist(z_query, z_memory, p=2)
    knn_dist, _ = torch.topk(dist, k=k, dim=1, largest=False)
    mean_dist = knn_dist.mean(dim=1)
    # Normalize by typical train distance and scale to [0, T]
    med = mean_dist.median().clamp(min=1e-8)
    return (mean_dist / med) * float(T)


def posterior_mean_from_recon(
    rec_per_t: torch.Tensor,
    timesteps: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Soft posterior E[t|x] from per-timestep reconstruction errors.
    p(t|x) ∝ rec_t(x); anomalies concentrate mass on harder (typically larger) t.
    rec_per_t: (B, K), timesteps: (K,)
    """
    if rec_per_t.numel() == 0:
        return torch.zeros(rec_per_t.size(0), device=rec_per_t.device)
    logits = rec_per_t / (temperature * rec_per_t.mean(dim=1, keepdim=True).clamp(min=1e-8))
    p = F.softmax(logits, dim=1)
    t_vals = timesteps.to(rec_per_t.device).float().view(1, -1)
    return (p * t_vals).sum(dim=1)


def fuse_dte_scores(
    knn_score: torch.Tensor,
    posterior_score: torch.Tensor,
    knn_weight: float = 0.5,
) -> torch.Tensor:
    """Combine kNN and reconstruction-softmax DTE estimates."""
    w = float(knn_weight)
    return w * knn_score + (1.0 - w) * posterior_score
