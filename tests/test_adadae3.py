"""Unit tests for AdaDDAE-3 modules."""
from __future__ import annotations

import torch

from src.models.adadae3 import (
    FluxHead,
    HeliosTimeMap,
    SoftRejectionEMA,
    apply_aegis,
    epoche_step,
    EpocheState,
    fit_aegis,
    flux_loss,
    fuse_argos_views,
    fuse_lynx_views,
    helios_select_score_grid,
    kairos_train_T,
    nexus_v2_loss,
    orbis_score,
    plexus_score,
    soft_rejection_weights,
    strata_consistency_loss,
)
from src.models.adadae import AdaDDAE
from src.models.danc import NoiseConfig


def test_helios_and_kairos():
    h = HeliosTimeMap()
    t = h(torch.rand(8), T=50)
    assert t.shape == (8,)
    assert int(t.min()) >= 1 and int(t.max()) <= 50
    grid = helios_select_score_grid(50, 8)
    assert 2 <= len(grid) <= 8
    assert kairos_train_T(0, 10, 50) <= kairos_train_T(9, 10, 50)


def test_orbis_plexus_strata():
    x = torch.randn(6, 12)
    xh = x + 0.1 * torch.randn_like(x)
    s = orbis_score(x, xh, top_k=4)
    assert s.shape == (6,)
    mem = torch.randn(30, 8)
    q = torch.randn(4, 8)
    assert plexus_score(q, mem, n_neighbors=5).shape == (4,)
    loss = strata_consistency_loss(torch.randn(4, 16), torch.randn(4, 16), scales=2)
    assert torch.isfinite(loss)


def test_argos_lynx_aegis_rdt():
    views = {
        "a": torch.rand(20),
        "b": torch.rand(20) * 2,
        "c": torch.rand(20) * 0.5,
    }
    f1 = fuse_argos_views(views)
    f2 = fuse_lynx_views(views)
    assert f1.shape == f2.shape == (20,)
    st = fit_aegis(f1, alpha=0.1)
    out = apply_aegis(f1, st)
    assert out.shape == (20,)
    ema = SoftRejectionEMA()
    w, ema = soft_rejection_weights(torch.rand(30), ema)
    assert w.min() >= 0.1 - 1e-6


def test_nexus_v2_flux_epoche():
    z1 = torch.randn(16, 8)
    z2 = z1 + 0.05 * torch.randn_like(z1)
    assert torch.isfinite(nexus_v2_loss(z1, z2))
    head = FluxHead(5, hidden=32, time_dim=8)
    x = torch.randn(8, 5)
    noise = torch.randn_like(x)
    t = torch.rand(8)
    assert torch.isfinite(flux_loss(head, x, noise, t))
    st = EpocheState()
    did, fac, boost = epoche_step(st, 1.0, patience=1)
    assert boost >= 0


def test_adadae3_smoke_fit():
    torch.manual_seed(0)
    x = torch.randn(40, 6)
    model = AdaDDAE(
        input_dim=6,
        hidden_dims=[32, 32],
        latent_dim=8,
        noise_config=NoiseConfig(20, "linear", 1e-4, 0.02, 4),
        epochs=2,
        batch_size=16,
        eval_every=1,
        early_stop_patience=5,
        use_scs=True,
        scs_max_timesteps=4,
        use_helios=True,
        use_kairos=True,
        use_orbis=True,
        use_plexus=True,
        use_geode=True,
        use_aether=False,
        use_mirage=True,
        mirage_draws=2,
        use_nexus_v2=True,
        nexus_loss_weight=0.01,
        use_rdt_v2=True,
        use_atlas=True,
        use_flux=True,
        flux_loss_weight=0.01,
        use_aegis=True,
        fusion_mode="argos",
        use_dte_view=True,
        dte_memory_size=64,
        use_rejection_training=True,
        rejection_warmup_epochs=1,
    )
    info = model.fit(x, x_val=x[:10], early_stop_metric="val_loss")
    scores = model.predict(x[:12])
    assert scores.shape == (12,)
    assert torch.isfinite(scores).all()
    assert info["early_stop_metric"] == "val_loss"
