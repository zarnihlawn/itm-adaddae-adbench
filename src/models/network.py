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
        use_atlas: bool = False,
        atlas_cond_dim: int = 24,
        atlas_film_hidden: int = 32,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512]
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.time_emb_dim = time_emb_dim
        self.time_emb_type = time_emb_type
        self.use_atlas = bool(use_atlas)
        self._atlas_cond: Optional[torch.Tensor] = None

        if time_emb_type == "learnable":
            self.timestep_embedding = nn.Linear(1, time_emb_dim)
            self.phasor_embedding = None
        elif time_emb_type == "phasor":
            from .adadae3.phasor import PhasorTimeEmbedding

            self.timestep_embedding = None
            self.phasor_embedding = PhasorTimeEmbedding(time_emb_dim)
        elif time_emb_type == "sinusoidal":
            self.timestep_embedding = None
            self.phasor_embedding = None
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

        self.atlas_film = None
        if self.use_atlas:
            from .adadae3.atlas import FilmGenerator

            self.atlas_film = FilmGenerator(atlas_cond_dim, atlas_film_hidden, latent_dim)

        self.apply(self._init_weights)

    def set_atlas_cond(self, cond: Optional[torch.Tensor]) -> None:
        self._atlas_cond = cond

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
        if self.time_emb_type == "phasor" and self.phasor_embedding is not None:
            return self.phasor_embedding(t)
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
        z = self.encoder(h)
        if self.atlas_film is not None and self._atlas_cond is not None:
            from .adadae3.atlas import film

            cond = self._atlas_cond
            if cond.dim() == 1:
                cond = cond.unsqueeze(0).expand(z.size(0), -1)
            gamma, beta = self.atlas_film(cond.to(z.device))
            z = film(z, gamma, beta)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x_t, t)
        x0_hat = self.decode(z)
        return x0_hat, z
