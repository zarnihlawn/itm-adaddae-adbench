"""Unit tests for AdaDDAE-2 modules."""
from __future__ import annotations

import numpy as np
import torch

from src.models.adadae2 import (
    ChronosHypernet,
    aether_path_energy,
    chronos_policy,
    fit_calix_weights,
    fuse_calix_views,
    geode_score,
    meta_to_phi,
    nexus_ssl_loss,
    vicreg_loss,
)
from src.models.danc import NoiseConfig, estimate_meta_features
from src.models.network import DiffusionBottleneckAE
from src.models.scheduler import DiffusionScheduler


def test_chronos_policy_shapes():
    X = np.random.randn(80, 6).astype(np.float32)
    meta = estimate_meta_features(X, contamination_mode="label_free")
    net = ChronosHypernet()
    cfg = chronos_policy(meta, "semi-supervised", hypernet=net, use_learned=True)
    assert isinstance(cfg, NoiseConfig)
    assert 20 <= cfg.num_timesteps <= 100
    phi = meta_to_phi(meta)
    assert phi.shape == (6,)


def test_geode_score_positive():
    mem = torch.randn(40, 8)
    q = torch.randn(5, 8)
    s = geode_score(q, mem, n_neighbors=8, n_components=2)
    assert s.shape == (5,)
    assert torch.isfinite(s).all()


def test_calix_weights_sum():
    views = {
        "reconstruction": torch.rand(32),
        "latent": torch.rand(32) * 0.5,
        "residual": torch.rand(32) * 0.2,
    }
    w = fit_calix_weights(views)
    assert abs(sum(w.values()) - 1.0) < 1e-5
    fused = fuse_calix_views(views, w)
    assert fused.shape == (32,)


def test_nexus_and_aether():
    model = DiffusionBottleneckAE(4, [16, 16], 8, time_emb_dim=4)
    sched = DiffusionScheduler(20, device=torch.device("cpu"), beta_start=1e-4, beta_end=0.02)
    x = torch.randn(8, 4)
    t0 = torch.ones(8, dtype=torch.long)
    loss = nexus_ssl_loss(model, x, t0, noise_std=0.05)
    assert torch.isfinite(loss)
    z1 = torch.randn(16, 8)
    assert torch.isfinite(vicreg_loss(z1, z1 + 0.01))
    energy = aether_path_energy(model, sched, x, [1, 5, 10], weights=[0.3, 0.3, 0.4])
    assert energy.shape == (8,)
