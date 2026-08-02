"""FULL-DTE — sharpened soft posterior over t + kNN (DTE-proxy, not full ICLR DTE).

Polarity matches baseline dte.posterior_mean_from_recon: higher recon error →
higher posterior mass → larger E[t] for anomalies.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..dte import fuse_dte_scores, knn_dte_score


def full_dte_posterior(
    rec_per_t: torch.Tensor,
    timesteps: torch.Tensor,
    temperature: float = 0.5,
    sharpen: float = 1.5,
) -> torch.Tensor:
    """
    Soft E[t|x] from per-timestep reconstruction errors (anomaly polarity).
    p(t|x) ∝ (rec_t / mean)^sharpen / temperature — same direction as baseline DTE.
    rec_per_t: (B, K) reconstruction norms (higher = worse).
    """
    if rec_per_t.numel() == 0:
        return torch.zeros(rec_per_t.size(0), device=rec_per_t.device)
    mean = rec_per_t.mean(dim=1, keepdim=True).clamp(min=1e-8)
    # Match dte.posterior_mean_from_recon polarity: high error → high mass
    logits = (rec_per_t / mean) * float(sharpen) / max(float(temperature), 1e-6)
    p = F.softmax(logits, dim=1)
    t_vals = timesteps.to(rec_per_t.device).float().view(1, -1)
    e_t = (p * t_vals).sum(dim=1)
    # Mild entropy bonus so flat posteriors are not under-scored
    ent = -(p * (p.clamp(min=1e-8).log())).sum(dim=1)
    return e_t + 0.05 * ent * float(timesteps.float().max().clamp(min=1.0))


def full_dte_fuse(
    rec_per_t: torch.Tensor,
    timesteps: torch.Tensor,
    z0: torch.Tensor,
    z_memory: torch.Tensor | None,
    knn: int = 5,
    T: int = 50,
    knn_weight: float = 0.4,
) -> torch.Tensor:
    post = full_dte_posterior(rec_per_t, timesteps)
    if z_memory is None or z_memory.numel() == 0:
        return post
    knn_s = knn_dte_score(z0, z_memory, k=knn, T=T)
    return fuse_dte_scores(knn_s, post, knn_weight=knn_weight)
