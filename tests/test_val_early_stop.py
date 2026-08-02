"""Phase-0 integrity: val carve + fit rejects test labels."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.datasets import carve_val_from_train
from src.models.adadae import AdaDDAE
from src.models.danc import NoiseConfig


def _tiny_noise() -> NoiseConfig:
    return NoiseConfig(
        num_timesteps=10,
        scheduler="linear",
        beta_start=1e-4,
        beta_end=0.02,
        time_emb_dim=4,
    )


def test_carve_val_from_train_disjoint():
    X = np.arange(100, dtype=np.float32).reshape(50, 2)
    y = np.zeros(50, dtype=np.float32)
    X_fit, X_val, y_fit, y_val = carve_val_from_train(X, y, val_fraction=0.2, random_state=0)
    assert X_fit.shape[0] + X_val.shape[0] == 50
    assert X_val.shape[0] >= 8
    # No overlapping row indices (content unique here)
    fit_set = set(map(tuple, X_fit.tolist()))
    val_set = set(map(tuple, X_val.tolist()))
    assert fit_set.isdisjoint(val_set)
    assert len(y_fit) == X_fit.shape[0]
    assert len(y_val) == X_val.shape[0]


def test_fit_rejects_test_labels():
    model = AdaDDAE(
        input_dim=4,
        hidden_dims=[16, 16],
        latent_dim=4,
        noise_config=_tiny_noise(),
        epochs=1,
        batch_size=8,
        eval_every=1,
        contrastive=False,
        use_scs=False,
        use_multiview=False,
        use_uncertainty_view=False,
        use_dte_view=False,
        use_rejection_training=False,
        device=torch.device("cpu"),
    )
    x = torch.randn(32, 4)
    x_val = torch.randn(8, 4)
    with pytest.raises(ValueError, match="no longer accepts x_test"):
        model.fit(x, x_val=x_val, x_test=torch.randn(8, 4), y_test=torch.zeros(8))


def test_fit_val_loss_early_stop():
    model = AdaDDAE(
        input_dim=4,
        hidden_dims=[16, 16],
        latent_dim=4,
        noise_config=_tiny_noise(),
        epochs=6,
        batch_size=8,
        eval_every=2,
        early_stop_patience=4,
        contrastive=False,
        use_scs=False,
        use_multiview=False,
        use_uncertainty_view=False,
        use_dte_view=False,
        use_rejection_training=False,
        device=torch.device("cpu"),
    )
    x = torch.randn(40, 4)
    x_val = torch.randn(10, 4)
    info = model.fit(x, x_val=x_val, early_stop_metric="val_loss")
    assert info["early_stop_metric"] == "val_loss"
    assert np.isfinite(info["best_val_metric"])
    assert info["best_pr_auc"] is None
    assert any("val_loss" in h for h in info["history"] if "val_loss" in h)
