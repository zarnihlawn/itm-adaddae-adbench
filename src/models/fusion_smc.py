"""SMC: SNR-Calibrated Multi-View Fusion (train-normal reliability weights)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch


VIEW_NAMES = ["reconstruction", "latent", "residual", "uncertainty", "diffusion_time"]


def estimate_view_reliability(
    view_tensors: Dict[str, torch.Tensor],
    timestep_weights: List[float],
    epsilon: float = 1e-6,
) -> Dict[str, float]:
    """
    r_v(t) = 1 / (Var_t(s_v) + eps) aggregated over timesteps.
    view_tensors: each value shape (n_samples, n_timesteps) or (n_samples,) for scalar views.
    """
    reliability: Dict[str, float] = {}
    for name, tensor in view_tensors.items():
        if tensor is None or tensor.numel() == 0:
            reliability[name] = 0.0
            continue
        t = tensor.float()
        if t.dim() == 1:
            var = t.var(unbiased=False).item() + epsilon
        else:
            var_per_t = t.var(dim=0, unbiased=False)
            w = torch.tensor(timestep_weights[: t.size(1)], dtype=torch.float32, device=t.device)
            w = w / w.sum().clamp_min(1e-12)
            var = (var_per_t * w).sum().item() + epsilon
        reliability[name] = 1.0 / var
    total = sum(reliability.values()) + epsilon
    return {k: v / total for k, v in reliability.items()}


def fuse_smc_views(
    rec: torch.Tensor,
    lat: torch.Tensor,
    res: torch.Tensor,
    var: Optional[torch.Tensor],
    dte: Optional[torch.Tensor],
    reliability: Dict[str, float],
    use_uncertainty: bool = False,
    use_dte: bool = False,
) -> torch.Tensor:
    """Fuse normalized per-sample view scores with SMC reliability weights."""

    def _norm(v: torch.Tensor) -> torch.Tensor:
        return v / v.mean().clamp_min(1e-12)

    parts = {
        "reconstruction": reliability.get("reconstruction", 0.5) * _norm(rec),
        "latent": reliability.get("latent", 0.2) * _norm(lat),
        "residual": reliability.get("residual", 0.1) * _norm(res),
    }
    if use_uncertainty and var is not None:
        parts["uncertainty"] = reliability.get("uncertainty", 0.1) * _norm(var)
    if use_dte and dte is not None:
        parts["diffusion_time"] = reliability.get("diffusion_time", 0.1) * _norm(dte)
    return sum(parts.values())


def collect_train_view_samples(
    score_fn,
    x_train: torch.Tensor,
    n_cal: int = 256,
    n_draws: int = 2,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Sample training normals for SMC calibration.
    score_fn(xb) -> (rec, lat, res, var, dte) per batch.
    """
    if device is not None and x_train.device != device:
        x_train = x_train.to(device)
    n = min(n_cal, x_train.size(0))
    idx = torch.randperm(x_train.size(0), device=x_train.device)[:n]
    xb = x_train[idx]
    rec_list, lat_list, res_list, var_list, dte_list = [], [], [], [], []
    for draw in range(n_draws):
        rec, lat, res, var, dte = score_fn(xb, score_seed=draw)
        rec_list.append(rec.unsqueeze(1))
        lat_list.append(lat.unsqueeze(1))
        res_list.append(res.unsqueeze(1))
        if var is not None:
            var_list.append(var.unsqueeze(1))
        if dte is not None:
            dte_list.append(dte.unsqueeze(1))
    out = {
        "reconstruction": torch.cat(rec_list, dim=1),
        "latent": torch.cat(lat_list, dim=1),
        "residual": torch.cat(res_list, dim=1),
    }
    if var_list:
        out["uncertainty"] = torch.cat(var_list, dim=1)
    if dte_list:
        out["diffusion_time"] = torch.cat(dte_list, dim=1)
    return out
