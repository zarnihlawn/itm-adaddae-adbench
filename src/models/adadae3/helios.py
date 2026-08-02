"""AdaDDAE-3 Wave 1: HELIOS — continuous SNR time reparameterization."""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class HeliosTimeMap(nn.Module):
    """Monotonic map from u~U(0,1) to discrete timestep via soft SNR schedule."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, u: torch.Tensor, T: int) -> torch.Tensor:
        # Positive softplus slope for monotonicity in expectation
        raw = self.net(u.unsqueeze(-1).float())
        s = torch.sigmoid(raw.squeeze(-1))
        t = 1 + (s * (T - 1)).clamp(0, T - 1)
        return t.round().long().clamp(1, T)


def sample_helios_t(helios: HeliosTimeMap, batch: int, T: int, device: torch.device) -> torch.Tensor:
    u = torch.rand(batch, device=device)
    return helios(u, T)


def helios_select_score_grid(T: int, k: int, alpha_bar: Optional[torch.Tensor] = None) -> List[int]:
    """Uniform in log-SNR space if alpha_bar given, else linspace."""
    k = max(2, min(k, T - 1))
    if alpha_bar is None or alpha_bar.numel() < T:
        return [int(x) for x in torch.linspace(1, T - 1, k).round().tolist()]
    # alpha_bar indexed 0..T-1 for t=1..T
    snr = (alpha_bar[: T - 1] / (1.0 - alpha_bar[: T - 1] + 1e-8)).clamp_min(1e-8)
    log_snr = torch.log(snr)
    targets = torch.linspace(log_snr[0].item(), log_snr[-1].item(), k)
    chosen = []
    for tgt in targets:
        idx = int(torch.argmin((log_snr - tgt).abs()).item()) + 1
        chosen.append(idx)
    return sorted(set(max(1, min(T - 1, c)) for c in chosen))
