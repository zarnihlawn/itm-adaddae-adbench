"""AXION model unit tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axion.models import build_model
from axion.models.mcb import sample_masks, scale_hparams
from axion.train.experiment import run_one_array


def test_scale_hparams_branches():
    small = scale_hparams(100, 10)
    big_d = scale_hparams(1000, 512)
    assert small["hidden"] <= big_d["hidden"]
    assert big_d["latent"] >= 64
    assert big_d["score_k"] >= 12
    assert big_d["high_mask_cap"] <= 0.6
    assert big_d["dropout"] <= 0.05


def test_sample_masks_shape():
    m = sample_masks(8, 16, rates=(0.2, 0.4))
    assert m.shape == (8, 16)
    assert ((m == 0) | (m == 1)).all()


def test_axion_fit_score_toy():
    rng = np.random.RandomState(0)
    X = rng.randn(120, 8).astype(np.float32)
    y = np.zeros(120, dtype=np.int64)
    y[-15:] = 1
    # anomalies = shift
    X[-15:] += 3.0
    model = build_model("axion", epochs=15, patience=5, batch_size=32, seed=0)
    model.fit(X[:100])  # mostly normals
    s = model.score(X)
    assert s.shape == (120,)
    assert np.all(np.isfinite(s))


def test_train_anchored_score_and_semi_latch():
    rng = np.random.RandomState(2)
    Xn = rng.randn(80, 8).astype(np.float32)
    Xa = rng.randn(20, 8).astype(np.float32) + 3.0
    X = np.vstack([Xn, Xa])
    y_all_normal = np.zeros(80, dtype=np.int64)

    model = build_model(
        "axion",
        epochs=12,
        patience=4,
        batch_size=32,
        seed=3,
        latch_alpha=0.4,
        latch_alpha_semi=0.25,
    )
    model.fit(Xn, y_all_normal)
    assert model.mcs_mean_ is not None
    assert model.mcs_std_ is not None and model.mcs_std_ > 0
    assert model.latch_score_mean_ is not None
    assert abs(model.active_latch_alpha_ - 0.25) < 1e-9

    s = model.score(X)
    assert s.shape == (100,)
    # Anomalies should rank higher on average under train-anchored scores
    assert float(s[80:].mean()) > float(s[:80].mean())

    # Unsupervised-like train (mixed labels) keeps full latch_alpha
    y_mixed = np.zeros(100, dtype=np.int64)
    y_mixed[80:] = 1
    model2 = build_model(
        "axion",
        epochs=8,
        patience=3,
        batch_size=32,
        seed=4,
        latch_alpha=0.4,
        latch_alpha_semi=0.25,
    )
    model2.fit(X, y_mixed)
    assert abs(model2.active_latch_alpha_ - 0.4) < 1e-9


def test_axion_run_one_array_smoke():
    rng = np.random.RandomState(1)
    X = rng.randn(100, 6).astype(np.float32)
    y = np.zeros(100, dtype=np.int64)
    y[-12:] = 1
    X[-12:] += 2.5
    r = run_one_array(
        X,
        y,
        dataset="toy",
        setting="semi-supervised",
        seed=111,
        model_name="axion",
        model_kwargs={"epochs": 12, "patience": 4, "batch_size": 32},
        protocol="paper",
    )
    assert np.isfinite(r.metrics["PR-AUC"])
    assert r.model == "axion"
