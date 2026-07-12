"""Selective Contaminated Scoring (SCS) — SNR-weighted timesteps."""
from __future__ import annotations

from typing import List, Sequence, Union

import numpy as np
import torch


def _linspace_timesteps(num_timesteps: int, setting: str, max_timesteps: int) -> List[int]:
    T = int(num_timesteps)
    if T <= 2:
        return [1]
    k = min(max_timesteps, T - 1)
    if setting == "unsupervised":
        start = max(1, T // 2)
        ts = np.linspace(start, T - 1, num=k)
    else:
        end = max(2, min(T - 1, max(T // 2, 100)))
        ts = np.linspace(1, end, num=k)
    ts = sorted(set(int(round(t)) for t in ts))
    ts = [t for t in ts if 1 <= t <= T - 1]
    return ts if ts else [1]


def _snr_stratified_timesteps(
    num_timesteps: int,
    setting: str,
    max_timesteps: int,
    alpha_bar: Union[np.ndarray, torch.Tensor, None] = None,
) -> List[int]:
    """SSTS: stratified quantile sampling on alpha_bar weighted by SNR importance."""
    T = int(num_timesteps)
    if T <= 2:
        return [1]
    k = min(max_timesteps, T - 1)
    candidates = list(range(1, T))

    if alpha_bar is not None:
        if isinstance(alpha_bar, torch.Tensor):
            ab = alpha_bar.detach().cpu().numpy()
        else:
            ab = np.asarray(alpha_bar)
        ab_t = np.array([ab[min(t, len(ab) - 1)] for t in candidates], dtype=np.float64)
        if setting == "unsupervised":
            weights = 1.0 - ab_t
        else:
            weights = ab_t
        weights = np.clip(weights, 1e-6, None)
        cum = np.cumsum(weights)
        cum = cum / (cum[-1] + 1e-12)
        targets = np.linspace(0, 1, k + 2)[1:-1]
        ts = []
        for tgt in targets:
            idx = int(np.searchsorted(cum, tgt))
            idx = min(idx, len(candidates) - 1)
            ts.append(candidates[idx])
        ts = sorted(set(ts))
        if len(ts) < k:
            extra = _linspace_timesteps(T, setting, k)
            ts = sorted(set(ts + extra))
        return ts[:k] if ts else [1]

    return _linspace_timesteps(T, setting, k)


def select_timesteps(
    num_timesteps: int,
    setting: str,
    max_timesteps: int = 32,
    mode: str = "snr_weighted",
    selection: str = "linspace",
    alpha_bar: Union[np.ndarray, torch.Tensor, None] = None,
) -> List[int]:
    T = int(num_timesteps)
    if T <= 2:
        return [1]

    if mode == "full_sum":
        return list(range(1, T))

    if selection == "snr_stratified":
        return _snr_stratified_timesteps(T, setting, max_timesteps, alpha_bar=alpha_bar)

    return _linspace_timesteps(T, setting, max_timesteps)


def timestep_weights(
    timesteps: Sequence[int],
    setting: str,
    T: int,
    mode: str = "snr_weighted",
    alpha_bar: Union[np.ndarray, torch.Tensor, None] = None,
) -> np.ndarray:
    ts = np.asarray(list(timesteps), dtype=np.float64)
    if mode == "uniform":
        w = np.ones(len(ts), dtype=np.float64)
    elif mode == "full_sum" or mode == "snr_weighted":
        if alpha_bar is not None:
            if isinstance(alpha_bar, torch.Tensor):
                ab = alpha_bar.detach().cpu().numpy()
            else:
                ab = np.asarray(alpha_bar)
            ab_t = np.array([ab[min(int(t), len(ab) - 1)] for t in ts], dtype=np.float64)
            if setting == "unsupervised":
                w = 1.0 - ab_t
            else:
                w = ab_t
            w = np.clip(w, 1e-6, None)
        else:
            if setting == "unsupervised":
                w = ts / max(T, 1)
            else:
                w = 1.0 - (ts / max(T, 1))
                w = np.clip(w, 0.1, 1.0)
    else:
        w = np.ones(len(ts), dtype=np.float64)

    w = w / (w.sum() + 1e-12)
    return w.astype(np.float64)


def vectorized_q_sample(
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    alpha_bar: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """
    x0: (B, d), timesteps: (K,), noise: (B, K, d)
    Returns x_t: (B, K, d)
    """
    ab = alpha_bar[timesteps - 1].view(1, -1, 1)
    return torch.sqrt(ab) * x0.unsqueeze(1) + torch.sqrt(1.0 - ab) * noise
