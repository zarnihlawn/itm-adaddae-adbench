"""Dataset-Adaptive Noise Controller (DANC) with SNR-guided scheduling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from .scheduler import DiffusionScheduler


@dataclass
class NoiseConfig:
    num_timesteps: int
    scheduler: str
    beta_start: float
    beta_end: float
    time_emb_dim: int
    tau_snr: Optional[float] = None
    contamination_est: Optional[float] = None
    contamination_mode: str = "label_free"


def _skewness(X: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    std = X.std(axis=0) + 1e-12
    m3 = ((X - mu) ** 3).mean(axis=0)
    return float(np.mean(np.abs(m3 / (std**3))))


def _intrinsic_dim_proxy(X: np.ndarray, max_samples: int = 2000) -> float:
    from sklearn.decomposition import PCA

    n, d = X.shape
    if n < 3 or d < 2:
        return float(d)
    rng = np.random.RandomState(0)
    if n > max_samples:
        idx = rng.choice(n, max_samples, replace=False)
        X = X[idx]
    k = min(32, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=k, svd_solver="randomized", random_state=0)
    pca.fit(X)
    ev = pca.explained_variance_ratio_ + 1e-12
    ev = ev / ev.sum()
    return float(np.exp(-np.sum(ev * np.log(ev))))


def estimate_contamination_label_free(X: np.ndarray, max_samples: int = 5000) -> float:
    """Train-only heuristic: robust z-score tail fraction, capped to [0.01, 0.3]."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n < 10:
        return 0.05
    rng = np.random.RandomState(0)
    if n > max_samples:
        idx = rng.choice(n, max_samples, replace=False)
        X = X[idx]
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0) + 1e-8
    z = np.abs(X - med) / (1.4826 * mad)
    row_score = z.max(axis=1)
    thresh = np.percentile(row_score, 95)
    rate = float((row_score > thresh).mean())
    return float(np.clip(max(rate, 0.01), 0.01, 0.3))


def estimate_meta_features(
    X: np.ndarray,
    y: np.ndarray | None = None,
    contamination_mode: str = "label_free",
) -> Dict[str, float]:
    n, d = X.shape
    if contamination_mode == "oracle" and y is not None:
        contam = float(y.mean())
    else:
        contam = estimate_contamination_label_free(X)
    return {
        "n": float(n),
        "d": float(d),
        "log_n": float(np.log1p(n)),
        "log_d": float(np.log1p(d)),
        "contamination": contam,
        "skewness": _skewness(X),
        "intrinsic_dim": _intrinsic_dim_proxy(X),
        "contamination_mode": contamination_mode,
    }


def _mans_beta_end(beta0: float, idim: float, d: float, skew: float) -> float:
    """MANS: beta_end* = beta0 * (1 + log(1 + d_eff/d)), skew-adjusted."""
    ratio = idim / max(d, 1.0)
    beta = beta0 * (1.0 + np.log1p(ratio))
    if skew >= 2.0:
        beta *= 1.15
    return float(np.clip(beta, beta0, 0.05))


def _mans_tau_snr(
    tau0: float,
    setting: str,
    meta: Dict[str, float],
    hardware_profile: str,
) -> float:
    """MANS: tau_snr* = tau0 * (1 + 1_semi * c_hat)."""
    n = meta["n"]
    c_hat = meta["contamination"]
    if setting == "unsupervised":
        if hardware_profile == "gpu":
            tau0 = 1e-4 if n >= 10_000 else 5e-4
        else:
            tau0 = 5e-4 if n >= 10_000 else 1e-3
        return float(tau0)
    tau = tau0 * (1.0 + c_hat)
    if meta["d"] >= 256:
        tau = max(tau, 0.12)
    return float(np.clip(tau, 0.05, 0.25))


def _resolve_T_from_snr(
    scheduler_name: str,
    beta_start: float,
    beta_end: float,
    tau_snr: float,
    T_max: int,
    device: torch.device,
    t_min: int = 5,
) -> int:
    """Find smallest T where alpha_bar_T <= tau_snr.

    If no T in [t_min, T_max] reaches the target (common for semi + linear
    schedules where ᾱ_T stays above τ), fail closed to T_max — not floor t_min.
    """
    t_min = max(5, int(t_min))
    T_max = max(t_min, int(T_max))
    lo, hi = t_min, T_max
    best: Optional[int] = None
    while lo <= hi:
        mid = (lo + hi) // 2
        sched = DiffusionScheduler(
            num_timesteps=mid,
            device=device,
            beta_start=beta_start,
            beta_end=beta_end,
            scheduler=scheduler_name,
        )
        alpha_T = float(sched.alpha_bar[-1].item())
        if alpha_T <= tau_snr:
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1
    if best is None:
        return T_max
    return max(t_min, min(best, T_max))


def danc_select(
    meta: Dict[str, float],
    setting: str,
    hardware_profile: str = "cpu",
    device: Optional[torch.device] = None,
    t_min: int = 5,
) -> NoiseConfig:
    n, d = meta["n"], meta["d"]
    skew = meta["skewness"]
    idim = meta["intrinsic_dim"]
    dev = device or torch.device("cpu")
    contam_mode = str(meta.get("contamination_mode", "label_free"))

    if setting == "unsupervised":
        scheduler = "cosine"
        beta_end = _mans_beta_end(0.02, idim, d, skew)
        time_emb = 4 if d >= 256 else 8
        T_max = 500 if hardware_profile == "gpu" else 200
        if n >= 100_000:
            T_max = 300 if hardware_profile == "gpu" else 80
        elif n >= 10_000:
            T_max = 400 if hardware_profile == "gpu" else 150
        if d >= 256:
            T_max = min(T_max, 200 if hardware_profile == "gpu" else 100)
        tau0 = 1e-4
    else:
        scheduler = "linear"
        beta_end = _mans_beta_end(0.02, idim, d, skew)
        time_emb = 8 if d < 256 else 4
        T_max = 100
        if d >= 256:
            T_max = 50
        elif n < 500:
            T_max = 30
        elif meta["contamination"] > 0.2:
            T_max = 50
        if idim > 20 and d > 64:
            T_max = min(100, T_max + 20)
        tau0 = 0.08

    tau = _mans_tau_snr(tau0, setting, meta, hardware_profile)
    T = _resolve_T_from_snr(
        scheduler, 1e-4, beta_end, tau, T_max, dev, t_min=int(t_min)
    )

    return NoiseConfig(
        num_timesteps=int(T),
        scheduler=scheduler,
        beta_start=1e-4,
        beta_end=beta_end,
        time_emb_dim=time_emb,
        tau_snr=tau,
        contamination_est=float(meta["contamination"]),
        contamination_mode=contam_mode,
    )


def danc_policy(
    meta: Dict[str, float],
    setting: str,
    hardware_profile: str = "cpu",
    device: Optional[torch.device] = None,
    t_min: int = 5,
) -> NoiseConfig:
    return danc_select(
        meta,
        setting,
        hardware_profile=hardware_profile,
        device=device,
        t_min=t_min,
    )
