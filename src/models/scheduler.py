"""Diffusion noise schedulers with correct beta/alpha_bar handling."""
from __future__ import annotations

import torch


class DiffusionScheduler:
    def __init__(
        self,
        num_timesteps: int,
        device: torch.device,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        scheduler: str = "linear",
    ):
        self.num_timesteps = int(num_timesteps)
        self.device = device
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.scheduler_name = scheduler

        self.beta = self._init_beta(scheduler).to(device).float()
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor):
        noise = torch.randn_like(x_0)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1.0 - alpha_bar_t) * noise
        return x_t, noise

    def _init_beta(self, scheduler: str) -> torch.Tensor:
        T = self.num_timesteps
        if scheduler == "linear":
            return torch.linspace(self.beta_start, self.beta_end, T)
        if scheduler == "quadratic":
            return torch.linspace(self.beta_start**0.5, self.beta_end**0.5, T) ** 2
        if scheduler == "cosine":
            # Nichol & Dhariwal cosine schedule -> derive betas from alpha_bar
            s = 0.008
            steps = torch.arange(T + 1, dtype=torch.float64)
            f = torch.cos((steps / T + s) / (1 + s) * torch.pi / 2) ** 2
            alpha_bar = (f / f[0]).float()
            betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
            return torch.clamp(betas, 1e-5, 0.999)
        if scheduler == "sigmoid":
            betas = torch.linspace(-6, 6, T)
            return torch.sigmoid(betas) * (self.beta_end - self.beta_start) + self.beta_start
        if scheduler == "exponential":
            return torch.logspace(
                torch.log10(torch.tensor(self.beta_start)),
                torch.log10(torch.tensor(self.beta_end)),
                T,
            )
        raise ValueError(f"Invalid scheduler: {scheduler}")
