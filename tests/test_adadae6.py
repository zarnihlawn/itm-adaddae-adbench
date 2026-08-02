"""Unit tests for AdaDDAE-6 modules."""
from __future__ import annotations

import numpy as np
import torch

from src.models.adadae import AdaDDAE
from src.models.adadae6 import (
    apex_transform,
    delta_refine_noise,
    delta_sandwich_contam,
    fit_apex,
    fit_kale_weights,
    fit_orbit,
    helix_refine_noise,
    kale_fuse,
    locus_lof_score,
    nautilus_model_dims,
    orbit_score,
    ridge_huber_loss,
    spiral_consistency_score,
    torque_memory_size,
)
from src.models.danc import NoiseConfig
from src.models.scheduler import DiffusionScheduler


def test_helix_and_delta():
    rng = np.random.RandomState(0)
    X = rng.randn(60, 8).astype(np.float32)
    noise = NoiseConfig(40, "linear", 1e-4, 0.02, 4, tau_snr=0.08, contamination_est=0.05)
    h = helix_refine_noise(noise, X, "semi-supervised", meta={"contamination": 0.05})
    assert h.scheduler in ("linear", "cosine")
    assert h.num_timesteps >= 5
    d = delta_refine_noise(h, "semi-supervised", meta={"contamination": 0.4})
    assert 0.005 <= float(d.contamination_est) <= 0.35
    assert delta_sandwich_contam(0.001, "unsupervised") <= 0.05


def test_apex_and_orbit_locus_spiral():
    rng = np.random.RandomState(1)
    raw = rng.exponential(1.0, 200)
    st = fit_apex(raw, contamination=0.05)
    assert st is not None
    out = apex_transform(torch.tensor(raw[:10], dtype=torch.float32), st)
    assert out.shape == (10,)
    ost = fit_orbit(rng.randn(40, 6).astype(np.float32))
    x = torch.randn(5, 6)
    xh = torch.randn(5, 6)
    assert orbit_score(x, xh, ost).shape == (5,)
    mem = torch.randn(30, 4)
    z = torch.randn(5, 4)
    assert locus_lof_score(z, mem, k=5).shape == (5,)
    sched = DiffusionScheduler(16, torch.device("cpu"), 1e-4, 0.02, "linear")

    class _M(torch.nn.Module):
        def forward(self, x, t):
            return x, x[:, :4]

    s = spiral_consistency_score(_M(), sched, torch.randn(4, 8), t_frac=0.5)
    assert s.shape == (4,)


def test_nautilus_torque_ridge_kale():
    h, lat = nautilus_model_dims(80, [512, 512], 32)
    assert max(h) < 512
    assert torque_memory_size(100_000, 4096) <= 2048
    x = torch.randn(8, 5)
    xh = torch.randn(8, 5)
    assert ridge_huber_loss(xh, x).shape == (8,)
    views = {"a": torch.rand(20), "b": torch.rand(20), "reconstruction": torch.rand(20)}
    w = fit_kale_weights(views)
    assert abs(sum(w.values()) - 1.0) < 1e-5
    assert kale_fuse(views, w).shape == (20,)


def test_adadae6_smoke_fit_predict():
    torch.manual_seed(0)
    x = torch.randn(48, 8)
    model = AdaDDAE(
        input_dim=8,
        hidden_dims=[32, 32],
        latent_dim=8,
        noise_config=NoiseConfig(16, "linear", 1e-4, 0.02, 4, tau_snr=0.08),
        epochs=2,
        batch_size=16,
        eval_every=1,
        use_scs=True,
        scs_max_timesteps=4,
        use_dte_view=True,
        use_multiview=True,
        fusion_mode="kale",
        use_kale=True,
        use_mahala=True,
        use_orbit=True,
        use_locus=True,
        use_spiral=True,
        use_ridge=True,
        use_apex=True,
        use_evt_tail=True,
        use_nautilus=True,
        n_train=48,
        dte_memory_size=32,
        meta_features={"n": 48, "contamination": 0.05, "d": 8},
    )
    model.fit(x, x_val=x[:12], early_stop_metric="val_loss")
    scores = model.predict(x[:10])
    assert scores.shape == (10,)
    assert torch.isfinite(scores).all()
