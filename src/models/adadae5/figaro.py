"""FIGARO — Fisher Information Geometry Adaptive Rate Optimization.

Train-only schedule refinement: estimate a data-dependent diffusion-path
energy proxy from train samples and pick (T*, beta_end*) near the peak of
I(t)/Cost(t).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from ..danc import NoiseConfig, _resolve_T_from_snr
from ..scheduler import DiffusionScheduler


def _fisher_proxy_curve(
    X: np.ndarray,
    scheduler_name: str,
    beta_start: float,
    beta_end: float,
    T: int,
    max_samples: int = 512,
    n_t: int = 12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Data-dependent path energy: E[||x_t - x_0||] / sqrt(1-ᾱ_t) / t."""
    rng = np.random.RandomState(0)
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    if n < 2:
        raise ValueError("FIGARO needs at least 2 train rows")
    if n > max_samples:
        X = X[rng.choice(n, max_samples, replace=False)]
    # Timesteps must be in {1,...,T} but q_sample indexes alpha_bar[t] with
    # t as 1-based label where alpha_bar has length T (indices 0..T-1 for t=1..T).
    # Clamp to T-1 max when T>1 to avoid edge IndexError on some schedulers.
    t_hi = max(1, T - 1) if T > 1 else 1
    ts = np.unique(np.linspace(1, t_hi, num=min(n_t, t_hi), dtype=int))
    ts = ts[(ts >= 1) & (ts <= t_hi)]
    if ts.size == 0:
        ts = np.array([1], dtype=int)

    device = torch.device("cpu")
    sched = DiffusionScheduler(
        num_timesteps=T,
        device=device,
        beta_start=beta_start,
        beta_end=beta_end,
        scheduler=scheduler_name,
    )
    xb = torch.tensor(X, dtype=torch.float32, device=device)
    scores = []
    for t_val in ts:
        t = torch.full((xb.size(0),), int(t_val), dtype=torch.long, device=device)
        x_t, _noise = sched.q_sample(xb, t)
        # Match DiffusionScheduler.q_sample indexing: alpha_bar[t] for t in 1..T-1
        ab_idx = int(np.clip(t_val, 0, sched.alpha_bar.numel() - 1))
        ab = float(sched.alpha_bar[ab_idx].item())
        # Data-dependent: displacement of corrupted train points (not ||ε|| alone)
        disp = torch.norm(x_t - xb, dim=1).mean().item()
        mag = disp / max(np.sqrt(max(1.0 - ab, 0.0)), 1e-6)
        scores.append(mag / max(float(t_val), 1.0))
    return np.asarray(scores, dtype=np.float64), ts.astype(np.float64)


def figaro_refine_noise(
    noise: NoiseConfig,
    X_train: np.ndarray,
    setting: str,
    meta: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
) -> NoiseConfig:
    """Refine DANC/NoiseConfig using data-dependent path-energy peak."""
    meta = meta or {}
    T0 = max(5, int(noise.num_timesteps))
    try:
        curve, ts = _fisher_proxy_curve(
            X_train,
            noise.scheduler,
            float(noise.beta_start),
            float(noise.beta_end),
            T0,
        )
    except (ValueError, RuntimeError, IndexError) as exc:
        print(f"FIGARO refine skipped: {exc}")
        return noise

    if curve.size == 0 or not np.isfinite(curve).any():
        print("FIGARO refine skipped: empty/non-finite curve")
        return noise

    peak = float(ts[int(np.nanargmax(curve))])
    if setting == "unsupervised":
        T_star = int(np.clip(peak * 1.35, 20, max(T0, 200)))
        beta_end = float(np.clip(noise.beta_end * (1.0 + 0.15 * (peak / max(T0, 1))), 1e-4, 0.05))
    else:
        T_star = int(np.clip(peak * 1.1, 15, max(T0, 100)))
        beta_end = float(np.clip(noise.beta_end * (1.0 + 0.08 * (peak / max(T0, 1))), 1e-4, 0.04))

    contam = float(meta.get("contamination", noise.contamination_est or 0.05))
    if contam > 0.15:
        T_star = max(10, int(T_star * 0.85))

    # Fisher peak floor — SNR resolve must not discard T★ by collapsing to 5.
    peak_floor = int(T_star)

    tau = float(noise.tau_snr) if noise.tau_snr is not None else (1e-3 if setting == "unsupervised" else 0.1)
    dev = device or torch.device("cpu")
    try:
        resolved = _resolve_T_from_snr(
            noise.scheduler, float(noise.beta_start), beta_end, tau, T_star, dev
        )
        # Keep Fisher peak as a lower bound so FIGARO actually changes depth.
        T_star = max(int(resolved), peak_floor)
    except (ValueError, RuntimeError, IndexError) as exc:
        print(f"FIGARO T* SNR resolve skipped: {exc}")
        T_star = peak_floor

    return NoiseConfig(
        num_timesteps=int(T_star),
        scheduler=noise.scheduler,
        beta_start=float(noise.beta_start),
        beta_end=float(beta_end),
        time_emb_dim=int(noise.time_emb_dim),
        tau_snr=noise.tau_snr,
        contamination_est=noise.contamination_est,
        contamination_mode=noise.contamination_mode,
    )
