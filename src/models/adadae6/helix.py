"""HELIX — train-only schedule family pick (linear vs cosine) via path energy."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from ..danc import NoiseConfig, _resolve_T_from_snr
from ..scheduler import DiffusionScheduler


def _path_energy(
    X: np.ndarray,
    scheduler_name: str,
    beta_start: float,
    beta_end: float,
    T: int,
    max_samples: int = 512,
) -> float:
    rng = np.random.RandomState(0)
    X = np.asarray(X, dtype=np.float32)
    if X.shape[0] < 2:
        return 0.0
    if X.shape[0] > max_samples:
        X = X[rng.choice(X.shape[0], max_samples, replace=False)]
    T = max(5, int(T))
    device = torch.device("cpu")
    sched = DiffusionScheduler(
        num_timesteps=T,
        device=device,
        beta_start=beta_start,
        beta_end=beta_end,
        scheduler=scheduler_name,
    )
    xb = torch.tensor(X, dtype=torch.float32, device=device)
    t_hi = max(1, T - 1)
    ts = np.unique(np.linspace(1, t_hi, num=min(8, t_hi), dtype=int))
    scores = []
    for t_val in ts:
        t = torch.full((xb.size(0),), int(t_val), dtype=torch.long, device=device)
        x_t, _ = sched.q_sample(xb, t)
        ab_idx = int(np.clip(t_val, 0, sched.alpha_bar.numel() - 1))
        ab = float(sched.alpha_bar[ab_idx].item())
        disp = torch.norm(x_t - xb, dim=1).mean().item()
        mag = disp / max(np.sqrt(max(1.0 - ab, 0.0)), 1e-6)
        scores.append(mag / max(float(t_val), 1.0))
    arr = np.asarray(scores, dtype=np.float64)
    peak = float(np.nanmax(arr)) if arr.size else 0.0
    # Prefer sharper peak / cost: peak / log(T)
    return peak / max(np.log(T + 1.0), 1.0)


def helix_refine_noise(
    noise: NoiseConfig,
    X_train: np.ndarray,
    setting: str,
    meta: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
) -> NoiseConfig:
    """Pick linear vs cosine by train path-energy proxy; keep T/β from DANC/FIGARO."""
    meta = meta or {}
    T0 = max(5, int(noise.num_timesteps))
    families = ["linear", "cosine"]
    # Semi often prefers linear; still allow cosine if energy wins clearly
    scores = {}
    for fam in families:
        try:
            scores[fam] = _path_energy(
                X_train, fam, float(noise.beta_start), float(noise.beta_end), T0
            )
        except (ValueError, RuntimeError, IndexError):
            scores[fam] = -1.0
    best = max(scores, key=scores.get)
    # Require clear margin to flip away from DANC default
    current = str(noise.scheduler)
    if best != current and scores[best] > scores.get(current, 0.0) * 1.05:
        chosen = best
    else:
        chosen = current if current in families else best

    tau = float(noise.tau_snr) if noise.tau_snr is not None else (
        1e-3 if setting == "unsupervised" else 0.1
    )
    dev = device or torch.device("cpu")
    try:
        T_star = _resolve_T_from_snr(
            chosen, float(noise.beta_start), float(noise.beta_end), tau, T0, dev
        )
        T_star = max(int(T_star), min(T0, 15))
    except (ValueError, RuntimeError, IndexError):
        T_star = T0

    return NoiseConfig(
        num_timesteps=int(T_star),
        scheduler=str(chosen),
        beta_start=float(noise.beta_start),
        beta_end=float(noise.beta_end),
        time_emb_dim=int(noise.time_emb_dim),
        tau_snr=noise.tau_snr,
        contamination_est=noise.contamination_est,
        contamination_mode=noise.contamination_mode,
    )
