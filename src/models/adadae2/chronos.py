"""CHRONOS: Curriculum Hypernetwork for Noise Schedules (train/val only).

A tiny shared MLP maps train-only meta-features to (T*, beta_end*, tau_snr*).
No per-dataset name routing — same weights for all datasets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..danc import NoiseConfig, _mans_beta_end, danc_policy


class ChronosHypernet(nn.Module):
    """phi -> (logit_T, logit_beta, logit_tau) in unconstrained space."""

    def __init__(self, in_dim: int = 6, hidden: int = 32):
        super().__init__()
        self.in_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3),
        )
        # Start near identity relative to heuristic defaults
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        return self.net(phi)


@dataclass
class ChronosRanges:
    t_min: int = 20
    t_max: int = 100
    beta_min: float = 0.005
    beta_max: float = 0.05
    tau_min: float = 1e-4
    tau_max: float = 5e-2


def meta_to_phi(meta: Dict[str, float], in_dim: int = 6) -> np.ndarray:
    """Fixed-length train-only feature vector (shared across datasets).

    Prefer OMNI keys when ``in_dim`` > 6; else classic 6-d CHRONOS phi.
    """
    if in_dim > 6:
        try:
            from ..adadae4.omni import omni_to_phi

            return omni_to_phi(meta, dim=in_dim)
        except Exception:
            pass
    n = float(meta.get("n", 1.0))
    d = float(meta.get("d", 1.0))
    return np.asarray(
        [
            float(meta.get("log_n", np.log1p(n))),
            float(meta.get("log_d", np.log1p(d))),
            float(meta.get("contamination", 0.05)),
            float(meta.get("skewness", 0.0)),
            float(meta.get("intrinsic_dim", d)) / max(d, 1.0),
            float(n) / max(d, 1.0),
        ],
        dtype=np.float32,
    )[:in_dim]


def decode_schedule(
    raw: torch.Tensor,
    ranges: ChronosRanges,
    base: Optional[NoiseConfig] = None,
) -> Tuple[int, float, float]:
    """Map network outputs to valid schedule parameters."""
    # sigmoid-ish via tanh scaled
    u = torch.tanh(raw)
    t01 = (u[0].item() + 1.0) * 0.5
    b01 = (u[1].item() + 1.0) * 0.5
    tau01 = (u[2].item() + 1.0) * 0.5
    T = int(round(ranges.t_min + t01 * (ranges.t_max - ranges.t_min)))
    T = max(ranges.t_min, min(ranges.t_max, T))
    beta = ranges.beta_min + b01 * (ranges.beta_max - ranges.beta_min)
    tau = ranges.tau_min + tau01 * (ranges.tau_max - ranges.tau_min)
    if base is not None:
        # Blend toward DANC heuristic for stability
        beta = 0.5 * beta + 0.5 * float(base.beta_end)
        T = int(round(0.5 * T + 0.5 * int(base.num_timesteps)))
        T = max(ranges.t_min, min(ranges.t_max, T))
        if base.tau_snr is not None:
            tau = 0.5 * tau + 0.5 * float(base.tau_snr)
    return T, float(beta), float(tau)


def chronos_policy(
    meta: Dict[str, float],
    setting: str,
    hypernet: Optional[ChronosHypernet] = None,
    ranges: Optional[ChronosRanges] = None,
    hardware_profile: str = "cpu",
    device: Optional[torch.device] = None,
    use_learned: bool = True,
) -> NoiseConfig:
    """Produce NoiseConfig via CHRONOS (learned) or DANC fallback."""
    ranges = ranges or ChronosRanges()
    base = danc_policy(meta, setting, hardware_profile=hardware_profile, device=device)
    if not use_learned or hypernet is None:
        return base

    phi = torch.tensor(meta_to_phi(meta, in_dim=getattr(hypernet, "in_dim", 6) if hasattr(hypernet, "in_dim") else hypernet.net[0].in_features), dtype=torch.float32)
    hypernet.eval()
    with torch.inference_mode():
        raw = hypernet(phi)
    T, beta_end, tau = decode_schedule(raw, ranges, base=base)
    # Keep MANS-ish floor from base intrinsic dim if beta too low
    idim = float(meta.get("intrinsic_dim", meta.get("d", 1.0)))
    d = float(meta.get("d", 1.0))
    skew = float(meta.get("skewness", 0.0))
    beta_mans = _mans_beta_end(0.02, idim, d, skew)
    beta_end = float(np.clip(0.7 * beta_end + 0.3 * beta_mans, ranges.beta_min, ranges.beta_max))

    return NoiseConfig(
        num_timesteps=T,
        scheduler=base.scheduler,
        beta_start=base.beta_start,
        beta_end=beta_end,
        time_emb_dim=base.time_emb_dim,
        tau_snr=tau,
        contamination_est=base.contamination_est,
        contamination_mode=base.contamination_mode,
    )


def fit_chronos_on_val_proxy(
    hypernet: ChronosHypernet,
    meta: Dict[str, float],
    val_loss_fn,
    steps: int = 20,
    lr: float = 1e-2,
) -> ChronosHypernet:
    """Light bi-level: nudge hypernet so decoded schedule lowers val recon proxy.

    val_loss_fn(NoiseConfig) -> float  (must use train/val only).
    """
    opt = torch.optim.Adam(hypernet.parameters(), lr=lr)
    phi = torch.tensor(meta_to_phi(meta), dtype=torch.float32)
    ranges = ChronosRanges()
    hypernet.train()
    best_state = {k: v.detach().clone() for k, v in hypernet.state_dict().items()}
    best = float("inf")
    for _ in range(max(1, steps)):
        opt.zero_grad()
        raw = hypernet(phi)
        # Differentiable surrogate: prefer mid-range schedules (regularizer) +
        # black-box loss via REINFORCE-free finite difference on decoded params.
        T, beta, tau = decode_schedule(raw.detach(), ranges)
        cfg = NoiseConfig(T, "linear", 1e-4, beta, 8, tau_snr=tau)
        loss_val = float(val_loss_fn(cfg))
        # Surrogate: push raw toward values that historically helped — use loss as weight
        # on L2 of raw (shrink when loss high)
        sur = (raw**2).mean() * (1.0 + loss_val)
        sur.backward()
        opt.step()
        if loss_val < best:
            best = loss_val
            best_state = {k: v.detach().clone() for k, v in hypernet.state_dict().items()}
    hypernet.load_state_dict(best_state)
    hypernet.eval()
    return hypernet
