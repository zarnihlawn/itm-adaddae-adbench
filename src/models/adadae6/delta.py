"""DELTA — LF contamination sandwich → τ_snr retune (train-only)."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from ..danc import NoiseConfig, _resolve_T_from_snr


def delta_sandwich_contam(
    c_hat: float,
    setting: str,
    c_lo: float = 0.005,
    c_hi: float = 0.35,
) -> float:
    """Clip LF estimate into a sane band; soft-shrink extremes for rare/heavy."""
    c = float(np.clip(c_hat, c_lo, c_hi))
    if setting == "unsupervised" and c_hat < 0.02:
        # Rare: don't over-inflate τ via LF floor noise
        c = float(np.clip(c_hat, c_lo, 0.05))
    if c_hat > 0.25:
        c = float(min(c, 0.30))
    return c


def delta_refine_noise(
    noise: NoiseConfig,
    setting: str,
    meta: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
) -> NoiseConfig:
    """Retune τ from sandwiched ĉ and optionally re-resolve T."""
    meta = meta or {}
    c_raw = float(
        meta.get("contamination", noise.contamination_est if noise.contamination_est is not None else 0.05)
    )
    c_tilde = delta_sandwich_contam(c_raw, setting)
    tau0 = 1e-4 if setting == "unsupervised" else 0.08
    # Mild contam → τ scaling (same spirit as MANS)
    tau = float(np.clip(tau0 * (1.0 + c_tilde), 0.05 if setting != "unsupervised" else 1e-4, 0.25))
    if setting == "unsupervised":
        tau = float(np.clip(tau0 * (1.0 + c_tilde), 1e-4, 1e-2))

    T0 = max(5, int(noise.num_timesteps))
    dev = device or torch.device("cpu")
    try:
        T_star = _resolve_T_from_snr(
            noise.scheduler, float(noise.beta_start), float(noise.beta_end), tau, T0, dev
        )
        T_star = max(int(T_star), min(T0, 10))
    except (ValueError, RuntimeError, IndexError):
        T_star = T0

    return NoiseConfig(
        num_timesteps=int(T_star),
        scheduler=noise.scheduler,
        beta_start=float(noise.beta_start),
        beta_end=float(noise.beta_end),
        time_emb_dim=int(noise.time_emb_dim),
        tau_snr=float(tau),
        contamination_est=float(c_tilde),
        contamination_mode=noise.contamination_mode,
    )
