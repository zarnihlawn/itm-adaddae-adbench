"""AEGIS — split-conformal calibration + temperature scaling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class AegisState:
    quantile: float
    temperature: float = 1.0
    offset: float = 0.0


@torch.inference_mode()
def fit_aegis(
    train_scores: torch.Tensor,
    alpha: float = 0.1,
    temperature: float = 1.0,
) -> AegisState:
    s = train_scores.float().reshape(-1)
    if s.numel() == 0:
        return AegisState(quantile=0.0, temperature=temperature)
    q = float(torch.quantile(s, 1.0 - alpha).item())
    return AegisState(quantile=q, temperature=max(1e-6, float(temperature)), offset=0.0)


@torch.inference_mode()
def apply_aegis(scores: torch.Tensor, state: Optional[AegisState]) -> torch.Tensor:
    if state is None:
        return scores
    # Soft exceedance relative to conformal band
    z = (scores.float() - state.quantile) / state.temperature
    return torch.nn.functional.softplus(z) + 1e-8
