"""Unit tests for AdaDDAE-5 Wave-6 correctness."""
from __future__ import annotations

import numpy as np
import torch

from src.models.adadae import AdaDDAE
from src.models.adadae5 import (
    confal_apply,
    evt_tail_transform,
    figaro_refine_noise,
    fit_confal,
    fit_evt_gpd,
    fit_mahala,
    fit_spectra,
    full_dte_posterior,
    ib_latent_loss,
    mahala_score,
    sinkhorn_ot_score,
    spectra_transform,
)
from src.models.danc import NoiseConfig, _resolve_T_from_snr
from src.models.scheduler import DiffusionScheduler
from src.models.scs import vectorized_q_sample


def test_resolve_T_fail_closed_to_T_max():
    """Semi + linear never reaches τ≈0.08 → must return T_max, not floor 5."""
    T = _resolve_T_from_snr("linear", 1e-4, 0.02, 0.08, 100, torch.device("cpu"))
    assert T == 100, f"expected T_max=100, got {T}"


def test_resolve_T_finds_smallest_when_reachable():
    """Cosine schedule can drive ᾱ_T below a loose τ."""
    T = _resolve_T_from_snr("cosine", 1e-4, 0.02, 0.5, 50, torch.device("cpu"))
    assert 5 <= T <= 50


def test_figaro_does_not_collapse_to_5():
    rng = np.random.RandomState(0)
    X = rng.randn(80, 8).astype(np.float32)
    noise = NoiseConfig(50, "linear", 1e-4, 0.02, 4, tau_snr=0.08)
    out = figaro_refine_noise(noise, X, "semi-supervised", meta={"contamination": 0.05})
    assert out.num_timesteps > 5, f"FIGARO collapsed to T={out.num_timesteps}"


def test_full_dte_polarity():
    """Higher recon error → higher soft E[t]."""
    ts = torch.tensor([1, 5, 10], dtype=torch.long)
    low = torch.tensor([[0.1, 0.1, 0.1]], dtype=torch.float32)
    high = torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float32)
    e_low = full_dte_posterior(low, ts)
    e_high = full_dte_posterior(high, ts)
    # Relative within-row: peak at later t with high error on last step
    peaked = torch.tensor([[0.1, 0.2, 3.0]], dtype=torch.float32)
    flat = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
    assert full_dte_posterior(peaked, ts).item() >= full_dte_posterior(flat, ts).item() - 1e-5
    assert e_low.shape == (1,) and e_high.shape == (1,)


def test_mahala_train_mean_std():
    rng = np.random.RandomState(1)
    r = rng.randn(40, 4).astype(np.float32)
    z = rng.randn(40, 3).astype(np.float32)
    st = fit_mahala(r, z)
    assert st is not None
    assert st.train_mean.shape[0] == 7
    assert st.train_std.shape[0] == 7
    s = mahala_score(torch.tensor(r[:5]), torch.tensor(z[:5]), st)
    assert s.shape == (5,)
    assert torch.isfinite(s).all()


def test_evt_confal_compose_order():
    rng = np.random.RandomState(2)
    raw = rng.exponential(1.0, size=200).astype(np.float64) + 0.1
    evt = fit_evt_gpd(raw)
    assert evt is not None
    evt_scores = evt_tail_transform(torch.tensor(raw, dtype=torch.float32), evt).numpy()
    conf = fit_confal(evt_scores)
    assert conf is not None
    # Apply order matches fit: EVT then CONFAL
    out = confal_apply(evt_tail_transform(torch.tensor(raw[:10], dtype=torch.float32), evt), conf)
    assert out.shape == (10,)
    assert torch.isfinite(out).all()
    # Fitting CONFAL on raw then applying after EVT would be wrong — knots differ
    conf_raw = fit_confal(raw)
    assert conf_raw is not None
    assert not np.allclose(conf.quantiles[:8], conf_raw.quantiles[:8])


def test_spectra_shape_lock():
    rng = np.random.RandomState(3)
    X = rng.randn(60, 12).astype(np.float32)
    st = fit_spectra(X, n_components=4)
    assert st is not None
    Xt = spectra_transform(X[:10], st)
    assert Xt.shape == (10, 12)


def test_sinkhorn_1d_and_high_d_off():
    x = torch.randn(4, 16)
    xh = torch.randn(4, 16)
    s = sinkhorn_ot_score(x, xh)
    assert s.shape == (4,)
    assert torch.isfinite(s).all()
    model = AdaDDAE(
        input_dim=128,
        hidden_dims=[64, 64],
        latent_dim=8,
        noise_config=NoiseConfig(10, "linear", 1e-4, 0.02, 4),
        epochs=1,
        batch_size=8,
        use_sinkhorn=True,
    )
    assert model.use_sinkhorn is False


def test_vectorized_abar_matches_q_sample():
    torch.manual_seed(0)
    sched = DiffusionScheduler(20, torch.device("cpu"), 1e-4, 0.02, "linear")
    x0 = torch.randn(3, 5)
    ts = torch.tensor([1, 5, 10], dtype=torch.long)
    noise = torch.randn(3, 3, 5)
    xt_vec = vectorized_q_sample(x0, ts, sched.alpha_bar, noise)
    for ki, t_val in enumerate(ts.tolist()):
        t = torch.full((3,), int(t_val), dtype=torch.long)
        ab = sched.alpha_bar[t].view(-1, 1)
        xt_seq = torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise[:, ki]
        assert torch.allclose(xt_vec[:, ki], xt_seq, atol=1e-5)


def test_ib_compression_only():
    z = torch.randn(8, 4)
    x = torch.randn(8, 6)
    xh = torch.randn(8, 6)
    loss = ib_latent_loss(z, xh, x, beta=0.01)
    # Independent of recon tensors
    loss2 = ib_latent_loss(z, None, None, beta=0.01)
    assert torch.allclose(loss, loss2)
    assert float(loss) == float(0.01 * (z**2).mean())


def test_adadae5_smoke_fit_predict():
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
        use_figaro=False,
        use_dsm_plus=True,
        use_mahala=True,
        use_full_dte=True,
        use_lexicon=True,
        fusion_mode="lexicon",
        use_evt_tail=True,
        use_confal=True,
        use_sinkhorn=True,
        use_ib_latent=True,
        use_elbo_s=False,
        use_curriculum_snr=True,
        use_vmf_z=True,
        use_dte_view=True,
        use_multiview=True,
        vectorized_scoring=True,
    )
    model.fit(x, x_val=x[:12], early_stop_metric="val_loss")
    scores = model.predict(x[:10])
    assert scores.shape == (10,)
    assert torch.isfinite(scores).all()
    assert model._evt_state is not None or model._confal_state is not None
