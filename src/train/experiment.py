"""Single experiment runner: one dataset file × setting × seed."""
from __future__ import annotations

import copy
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from ..data.datasets import load_npz, split_data
from ..eval.metrics import evaluate_anomaly_detection
from ..features.pipeline import FeatureTuningPipeline, infer_policy
from ..features.modality_encoder import ModalityEncoder, setting_blocks_mce
from ..memory import (
    apply_thread_limits,
    choose_score_batch_size,
    choose_train_batch_size,
    cleanup_memory,
    create_guard,
    guard_memory_mb,
    setup_cuda,
)
from ..models.adadae import AdaDDAE
from ..models.danc import NoiseConfig, danc_policy, estimate_meta_features
from ..policy import apply_routed_config, policy_overrides
from ..runlog.logger import RunLogger


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _fit_baseline_ddae(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Dict[str, Any],
    setting: str,
    seed: int,
    device: torch.device,
    guard,
) -> AdaDDAE:
    """Train DDAE-faithful baseline for GATE comparison."""
    cfg = _deep_update(config, policy_overrides("baseline_ddae"))
    cfg.setdefault("adadae", {})["use_gate"] = False
    cfg.setdefault("adadae", {})["use_mce"] = False
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    diff_cfg = cfg.get("diffusion", {})
    hw = cfg.get("hardware", {})
    noise = NoiseConfig(
        num_timesteps=int(diff_cfg.get("num_timesteps", 50)),
        scheduler=str(diff_cfg.get("scheduler", "linear")),
        beta_start=float(diff_cfg.get("beta_start", 1e-4)),
        beta_end=float(diff_cfg.get("beta_end", 0.02)),
        time_emb_dim=int(diff_cfg.get("time_emb_dim", 4)),
    )
    n = X_train.shape[0]
    batch_size = choose_train_batch_size(
        n,
        max_batch=int(hw.get("train_batch_size_max", 512)),
        large_n_threshold=int(hw.get("large_n_threshold", 100_000)),
    )
    score_bs = choose_score_batch_size(
        X_test.shape[0],
        max_batch=int(hw.get("score_batch_size_max", 1024)),
        large_n_threshold=int(hw.get("large_n_threshold", 100_000)),
        large_batch=int(hw.get("score_batch_size_large_n", 256)),
    )
    model = AdaDDAE(
        input_dim=X_train.shape[1],
        hidden_dims=list(model_cfg.get("hidden_dims", [512, 512])),
        latent_dim=int(model_cfg.get("latent_dim", 32)),
        activation=str(model_cfg.get("activation", "lrelu")),
        noise_config=noise,
        time_emb_type=str(diff_cfg.get("time_emb_type", "sinusoidal")),
        epochs=int(train_cfg.get("epochs", 100)),
        batch_size=batch_size,
        learning_rate=float(train_cfg.get("lr", 1e-3)),
        device=device,
        eval_every=int(train_cfg.get("eval_every", 10)),
        contrastive=False,
        use_scs=False,
        use_multiview=False,
        use_uncertainty_view=False,
        use_dte_view=False,
        use_rejection_training=False,
        fusion_mode="fixed",
        setting=setting,
        score_batch_size=score_bs,
        memory_guard=guard,
        early_stop_patience=int(train_cfg.get("early_stop_patience", 20)),
        use_amp=bool(hw.get("use_amp", False)),
        amp_dtype=str(hw.get("amp_dtype", "bfloat16")),
        pin_memory=bool(hw.get("pin_memory", False)),
        num_workers=int(hw.get("dataloader_num_workers", 0)),
        vectorized_scoring=bool(hw.get("vectorized_scoring", False)),
        preupload_test_threshold=int(hw.get("preupload_test_threshold", 50000)),
    )
    x_train_t = torch.tensor(X_train, dtype=torch.float32)
    x_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    model.fit(x_train_t, x_test_t, y_test_t, eval_fn=evaluate_anomaly_detection)
    return model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_single_file(
    npz_path: Path,
    setting: str,
    seed: int,
    config: Dict[str, Any],
    logger: Optional[RunLogger] = None,
    dataset_name: str = "",
    split_name: str = "",
    category: str = "classical",
) -> Dict[str, Any]:
    t_total_start = time.perf_counter()
    hw = config["hardware"]
    apply_thread_limits(hw.get("num_threads", 8))
    set_seed(seed)

    device_name = str(hw.get("device", "cpu"))
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
        hw = dict(hw)
        hw["device"] = "cpu"
        hw["use_amp"] = False
        hw["pin_memory"] = False
        hw["vectorized_scoring"] = False
        config = dict(config)
        config["hardware"] = hw
    device = torch.device(device_name)
    setup_cuda(hw)

    X, y = load_npz(npz_path)
    X_train, X_test, y_train, y_test = split_data(X, y, train_setting=setting, random_state=seed)

    max_train = int(hw.get("max_train_samples", 0) or 0)
    if max_train > 0 and X_train.shape[0] > max_train:
        rng = np.random.RandomState(seed)
        idx = rng.choice(X_train.shape[0], size=max_train, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]

    # v3 routing uses train shape meta before FTP/DANC
    route_meta = {
        "n": float(X_train.shape[0]),
        "d": float(X_train.shape[1]),
    }
    config = apply_routed_config(
        config, setting, category, dataset_name=dataset_name, meta=route_meta
    )
    config = dict(config)
    config["_setting"] = setting
    adadae_cfg = config.get("adadae", {})
    feat_cfg = config.get("features", {})
    use_ftp = bool(adadae_cfg.get("use_ftp", True))
    use_mce = bool(adadae_cfg.get("use_mce", False))

    t_ftp_start = time.perf_counter()
    mce_summary: Dict[str, Any] = {"use_mce": False}
    if use_mce and not setting_blocks_mce(config, category):
        mce_modality = str(adadae_cfg.get("mce_modality", category))
        enc = ModalityEncoder(modality=mce_modality)
        X_train = enc.fit_transform(X_train, seed=seed)
        X_test = enc.transform(X_test)
        mce_summary = enc.summary()
        mce_summary["use_mce"] = True

    if use_ftp:
        policy = infer_policy(
            n_samples=X_train.shape[0],
            n_features=X_train.shape[1],
            scaler=feat_cfg.get("scaler", "auto"),
            pca_dim_threshold=int(feat_cfg.get("pca_dim_threshold", 128)),
            pca_max_components=int(feat_cfg.get("pca_max_components", 128)),
            pca_variance=float(feat_cfg.get("pca_variance", 0.95)),
            clip_outliers=bool(feat_cfg.get("clip_outliers", True)),
            clip_sigma=float(feat_cfg.get("clip_sigma", 5.0)),
        )
        ftp = FeatureTuningPipeline(policy)
        X_train = ftp.fit_transform(X_train)
        X_test = ftp.transform(X_test)
        ftp_summary = ftp.summary()
    else:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)
        ftp_summary = {"scaler": "standard", "use_pca": False}
    ftp_sec = time.perf_counter() - t_ftp_start

    contam_mode = str(adadae_cfg.get("danc_contamination_mode", "label_free"))
    y_for_meta = None
    if contam_mode == "oracle":
        if setting == "unsupervised":
            y_for_meta = y_train
        else:
            y_for_meta = y

    meta = estimate_meta_features(X_train, y_for_meta, contamination_mode=contam_mode)
    if setting == "semi-supervised" and contam_mode == "oracle":
        meta["contamination"] = float(y.mean())

    diff_cfg = config.get("diffusion", {})
    hw_profile = str(hw.get("hardware_profile", "cpu"))
    if adadae_cfg.get("use_danc", True):
        noise = danc_policy(meta, setting, hardware_profile=hw_profile, device=device)
    else:
        noise = NoiseConfig(
            num_timesteps=int(diff_cfg.get("num_timesteps", 50)),
            scheduler=str(diff_cfg.get("scheduler", "linear")),
            beta_start=float(diff_cfg.get("beta_start", 1e-4)),
            beta_end=float(diff_cfg.get("beta_end", 0.02)),
            time_emb_dim=int(diff_cfg.get("time_emb_dim", 4)),
        )

    train_cfg = config.get("train", {})
    model_cfg = config.get("model", {})
    n = X_train.shape[0]
    batch_size = choose_train_batch_size(
        n,
        max_batch=int(hw.get("train_batch_size_max", 512)),
        large_n_threshold=int(hw.get("large_n_threshold", 100_000)),
    )
    score_bs = choose_score_batch_size(
        X_test.shape[0],
        max_batch=int(hw.get("score_batch_size_max", 1024)),
        large_n_threshold=int(hw.get("large_n_threshold", 100_000)),
        large_batch=int(hw.get("score_batch_size_large_n", 256)),
    )

    guard = create_guard(hw, device)

    scs_mode = str(adadae_cfg.get("scs_mode", "snr_weighted"))
    if adadae_cfg.get("scs_full_sum_ablation", False):
        scs_mode = "full_sum"

    if logger:
        logger.log(
            "job_start",
            dataset=dataset_name,
            split=split_name,
            setting=setting,
            seed=seed,
            n_train=int(X_train.shape[0]),
            n_test=int(X_test.shape[0]),
            d=int(X_train.shape[1]),
            batch_size=batch_size,
            noise=noise.__dict__,
            ftp=ftp_summary,
            mce=mce_summary,
            contamination_mode=contam_mode,
            contamination_est=meta.get("contamination"),
            device=str(device),
            rss_mb=guard.rss_mb() if hasattr(guard, "rss_mb") else None,
            vram_mb=guard.vram_mb() if hasattr(guard, "vram_mb") else None,
        )

    x_train_t = torch.tensor(X_train, dtype=torch.float32)
    x_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    contrastive = bool(train_cfg.get("contrastive", True))
    if setting == "unsupervised":
        c_alpha = float(train_cfg.get("contrastive_alpha_unsupervised", 0.05))
    else:
        c_alpha = float(train_cfg.get("contrastive_alpha", 0.2))

    model = AdaDDAE(
        input_dim=X_train.shape[1],
        hidden_dims=list(model_cfg.get("hidden_dims", [512, 512])),
        latent_dim=int(model_cfg.get("latent_dim", 32)),
        activation=str(model_cfg.get("activation", "lrelu")),
        noise_config=noise,
        time_emb_type=str(diff_cfg.get("time_emb_type", "sinusoidal")),
        epochs=int(train_cfg.get("epochs", 100)),
        batch_size=batch_size,
        learning_rate=float(train_cfg.get("lr", 1e-3)),
        device=device,
        eval_every=int(train_cfg.get("eval_every", 10)),
        contrastive=contrastive,
        contrastive_alpha=c_alpha,
        contrastive_gamma=float(train_cfg.get("contrastive_gamma", 1.0)),
        contrastive_adaptive_alpha=bool(train_cfg.get("contrastive_adaptive_alpha", True)),
        contrastive_pairing=str(adadae_cfg.get("contrastive_pairing", "taps")),
        hard_negative_mining=bool(train_cfg.get("hard_negative_mining", True)),
        use_scs=bool(adadae_cfg.get("use_scs", True)),
        scs_max_timesteps=int(adadae_cfg.get("scs_max_timesteps", 32)),
        scs_mode=scs_mode,
        scs_selection=str(adadae_cfg.get("scs_selection", "snr_stratified")),
        use_multiview=bool(adadae_cfg.get("use_multiview", True)),
        use_uncertainty_view=bool(adadae_cfg.get("use_uncertainty_view", False)),
        uncertainty_draws=int(adadae_cfg.get("uncertainty_draws", 3)),
        use_dte_view=bool(adadae_cfg.get("use_dte_view", True)),
        dte_knn=int(adadae_cfg.get("dte_knn", 5)),
        dte_memory_size=int(adadae_cfg.get("dte_memory_size", 4096)),
        dte_knn_weight=float(adadae_cfg.get("dte_knn_weight", 0.5)),
        use_rejection_training=bool(adadae_cfg.get("use_rejection_training", True)),
        rejection_quantile=float(adadae_cfg.get("rejection_quantile", 0.95)),
        rejection_min_weight=float(adadae_cfg.get("rejection_min_weight", 0.1)),
        rejection_warmup_epochs=int(adadae_cfg.get("rejection_warmup_epochs", 1)),
        fusion_mode=str(adadae_cfg.get("fusion_mode", "fixed")),
        fusion_weights=dict(adadae_cfg.get("fusion_weights", {})),
        setting=setting,
        score_batch_size=score_bs,
        memory_guard=guard,
        early_stop_patience=int(train_cfg.get("early_stop_patience", 20)),
        use_amp=bool(hw.get("use_amp", False)),
        amp_dtype=str(hw.get("amp_dtype", "bfloat16")),
        pin_memory=bool(hw.get("pin_memory", False)),
        num_workers=int(hw.get("dataloader_num_workers", 0)),
        vectorized_scoring=bool(hw.get("vectorized_scoring", False)),
        preupload_test_threshold=int(hw.get("preupload_test_threshold", 50000)),
        y_train=y_train_t,
    )

    fit_info = model.fit(
        x_train_t,
        x_test_t,
        y_test_t,
        logger=logger,
        eval_fn=evaluate_anomaly_detection,
    )

    scores = model.predict(x_test_t)
    metrics = evaluate_anomaly_detection(scores.detach().cpu().numpy(), y_test)

    gate_summary: Dict[str, Any] = {"use_gate": False}
    if adadae_cfg.get("use_gate", False):
        from ..ensemble.gate import (
            build_train_normal_scores,
            fit_isolation_forest,
            gate_winner_predict,
            isolation_scores,
            knn_dte_proxy_scores,
        )

        baseline_model = _fit_baseline_ddae(
            X_train, X_test, y_test, config, setting, seed, device, guard
        )

        def ad_fn(X: np.ndarray) -> np.ndarray:
            t = torch.tensor(X, dtype=torch.float32)
            return model.predict(t).detach().cpu().numpy()

        def ddae_fn(X: np.ndarray) -> np.ndarray:
            t = torch.tensor(X, dtype=torch.float32)
            return baseline_model.predict(t).detach().cpu().numpy()

        train_scores = build_train_normal_scores(X_train, ad_fn, ddae_fn, seed=seed)
        if_clf, if_scaler = fit_isolation_forest(X_train, seed=seed)
        if_test = isolation_scores(if_clf, if_scaler, X_test)
        knn_test = knn_dte_proxy_scores(X_train, X_test)
        ada_test = scores.detach().cpu().numpy()
        ddae_test = ddae_fn(X_test)
        test_scores = {
            "adadae": ada_test,
            "ddae": ddae_test,
            "iforest": if_test,
            "knn_dte": knn_test,
        }
        fused, decision = gate_winner_predict(
            test_scores,
            train_scores,
        )
        metrics = evaluate_anomaly_detection(fused, y_test)
        gate_summary = {
            "use_gate": True,
            "mode": "winner_take_all",
            "winner": decision.winner,
            "fallback": decision.fallback,
            "disagreement": decision.disagreement,
            "weights": decision.weights,
        }
        del baseline_model
        cleanup_memory(device)

    mem = guard_memory_mb(guard)
    train_sec = fit_info.get("timing", {}).get("train_sec", 0.0)
    score_sec = model.timing.get("score_sec", 0.0)
    total_sec = time.perf_counter() - t_total_start

    result = {
        "dataset": dataset_name,
        "split": split_name,
        "setting": setting,
        "seed": seed,
        "metrics": metrics,
        "noise": noise.__dict__,
        "ftp": ftp_summary,
        "mce": mce_summary,
        "gate": gate_summary,
        "best_pr_auc": fit_info.get("best_pr_auc"),
        "rss_mb": guard.rss_mb() if hasattr(guard, "rss_mb") else None,
        "vram_mb": mem if device.type == "cuda" else None,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "d": int(X_train.shape[1]),
        "score_timesteps": len(model.score_timesteps),
        "device": str(device),
        "contamination_mode": contam_mode,
        "contamination_est": meta.get("contamination"),
        "resolved_policy": adadae_cfg.get("resolved_policy"),
        "ftp_sec": ftp_sec,
        "train_sec": train_sec,
        "score_sec": score_sec,
        "total_sec": total_sec,
    }

    if logger:
        logger.log("job_end", **{k: v for k, v in result.items() if v is not None and k != "ftp"}, ftp=ftp_summary)

    del model, x_train_t, x_test_t, X_train, X_test, scores
    cleanup_memory(device)
    return result
