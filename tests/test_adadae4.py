"""Unit tests for AdaDDAE-4 regime modules."""
from __future__ import annotations

import numpy as np
import torch

from src.models.adadae4 import (
    enrich_omni_meta,
    fit_polis_prototypes,
    fit_prism,
    nano_model_dims,
    needle_tail_boost,
    omni_to_phi,
    polis_score,
    prism_score,
    quell_fuse,
    quell_reliability,
    regime_gate_flags,
    robust_normalize_views,
    sieve_rejection_quantile,
    sparse_score,
    fit_sparse_prototype,
    torrent_memory_size,
)
from src.models.adadae import AdaDDAE
from src.models.danc import NoiseConfig, estimate_meta_features


def test_omni_and_gates():
    X = np.random.randn(100, 10).astype(np.float32)
    meta = estimate_meta_features(X)
    meta = enrich_omni_meta(meta, X=X, category="classical")
    phi = omni_to_phi(meta, dim=16)
    assert phi.shape == (16,)
    gates = regime_gate_flags(meta)
    assert "use_nano" in gates


def test_nano_torrent_prism_polis():
    h, lat = nano_model_dims(80, [512, 512], 32)
    assert max(h) <= 128
    assert torrent_memory_size(100_000) <= 2048
    X = np.random.randn(200, 64).astype(np.float32)
    st = fit_prism(X, max_components=16)
    s = prism_score(torch.randn(5, 64), st)
    assert s.shape == (5,)
    proto = fit_polis_prototypes(np.random.randn(100, 8), cluster_sep=2.0)
    assert polis_score(torch.randn(4, 8), proto).shape == (4,)


def test_sieve_needle_sparse_quell():
    assert sieve_rejection_quantile(0.3) < 0.95
    s = needle_tail_boost(torch.rand(50))
    assert s.shape == (50,)
    proto = fit_sparse_prototype(np.random.rand(40, 20) * (np.random.rand(40, 20) > 0.7))
    assert sparse_score(torch.rand(5, 20), proto).shape == (5,)
    views = {"a": torch.rand(20), "b": torch.rand(20)}
    views = robust_normalize_views(views)
    rel = quell_reliability(views)
    assert abs(sum(rel.values()) - 1.0) < 1e-5
    assert quell_fuse(views, rel).shape == (20,)


def test_adadae4_smoke_fit():
    torch.manual_seed(0)
    x = torch.randn(50, 8)
    meta = enrich_omni_meta(estimate_meta_features(x.numpy()), X=x.numpy(), category="classical")
    model = AdaDDAE(
        input_dim=8,
        hidden_dims=[64, 64],
        latent_dim=8,
        noise_config=NoiseConfig(16, "linear", 1e-4, 0.02, 4),
        epochs=2,
        batch_size=16,
        eval_every=1,
        use_scs=True,
        scs_max_timesteps=4,
        use_helios=True,
        use_geode=True,
        use_omni=True,
        use_nano=True,
        use_prism=True,
        use_polis=True,
        use_sieve=True,
        use_robust=True,
        use_quell=True,
        fusion_mode="quell",
        auto_regime_gates=True,
        meta_features=meta,
        n_train=50,
        use_dte_view=True,
        dte_memory_size=32,
        use_rejection_training=True,
        rejection_warmup_epochs=1,
    )
    info = model.fit(x, x_val=x[:10], early_stop_metric="val_loss")
    scores = model.predict(x[:8])
    assert scores.shape == (8,)
    assert torch.isfinite(scores).all()
    assert info["early_stop_metric"] == "val_loss"
