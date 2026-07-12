"""Bottleneck encoder-decoder with timestep conditioning."""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


def _act(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    if name == "lrelu":
        return nn.LeakyReLU(0.2)
    raise ValueError(f"Unknown activation: {name}")


class DiffusionBottleneckAE(nn.Module):
    """Encoder (x_t ⊕ e_t) -> z -> decoder -> x0_hat."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        latent_dim: int = 32,
        time_emb_dim: int = 8,
        activation: str = "lrelu",
        time_emb_type: str = "sinusoidal",
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512]
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.time_emb_dim = time_emb_dim
        self.time_emb_type = time_emb_type

        if time_emb_type == "learnable":
            self.timestep_embedding = nn.Linear(1, time_emb_dim)
        elif time_emb_type == "sinusoidal":
            self.timestep_embedding = None
        else:
            raise ValueError(f"Invalid time_emb_type: {time_emb_type}")

        enc_in = input_dim + (time_emb_dim if time_emb_dim > 0 else 0)
        enc_layers: List[nn.Module] = []
        prev = enc_in
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), _act(activation)]
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: List[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), _act(activation)]
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def embed_t(self, t: torch.Tensor) -> torch.Tensor:
        if self.time_emb_dim <= 0:
            return t.new_zeros((t.shape[0], 0))
        if self.time_emb_type == "learnable":
            return self.timestep_embedding(t.unsqueeze(1).float())
        return self.sine_cosine_transform_timesteps(t.float())

    def sine_cosine_transform_timesteps(self, timesteps: torch.Tensor, max_period: int = 10000):
        dim_out = self.time_emb_dim
        half = dim_out // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, device=timesteps.device, dtype=torch.float32) / max(half, 1)
        )
        args = timesteps[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim_out % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def encode(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.time_emb_dim > 0:
            e_t = self.embed_t(t)
            h = torch.cat([x_t, e_t], dim=1)
        else:
            h = x_t
        return self.encoder(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x_t, t)
        x0_hat = self.decode(z)
        return x0_hat, z
