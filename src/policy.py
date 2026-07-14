"""Setting- and family-aware AdaDDAE v2 policy routing."""
from __future__ import annotations

import copy
from typing import Any, Dict

# DDAE-faithful baseline (semi classical): matches configs/baselines_ddae.yaml
BASELINE_DDAE: Dict[str, Any] = {
    "train": {
        "contrastive": False,
        "contrastive_alpha": 0.0,
        "hard_negative_mining": False,
    },
    "diffusion": {
        "num_timesteps": 50,
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "scheduler": "linear",
        "time_emb_dim": 4,
        "time_emb_type": "sinusoidal",
    },
    "adadae": {
        "use_danc": False,
        "use_scs": False,
        "use_ftp": False,
        "use_multiview": False,
        "use_uncertainty_view": False,
        "use_dte_view": False,
        "use_rejection_training": False,
        "fusion_mode": "fixed",
        "fusion_weights": {
            "reconstruction": 1.0,
            "latent": 0.0,
            "residual": 0.0,
            "uncertainty": 0.0,
            "diffusion_time": 0.0,
        },
    },
    "features": {
        "scaler": "standard",
        "pca_dim_threshold": 99999,
        "clip_outliers": False,
    },
}

# Unsupervised: LF-DANC + MANS + SSTS (ablation ladder winner)
UNSUP_SSTS: Dict[str, Any] = {
    "train": {
        "contrastive": False,
        "contrastive_alpha": 0.0,
        "hard_negative_mining": False,
    },
    "adadae": {
        "use_danc": True,
        "use_scs": True,
        "use_ftp": True,
        "use_multiview": False,
        "scs_mode": "snr_weighted",
        "scs_selection": "snr_stratified",
        "scs_max_timesteps": 50,
        "danc_contamination_mode": "label_free",
        "use_uncertainty_view": False,
        "use_dte_view": False,
        "use_rejection_training": False,
        "contrastive_pairing": "random",
        "fusion_mode": "fixed",
        "fusion_weights": {
            "reconstruction": 1.0,
            "latent": 0.0,
            "residual": 0.0,
            "uncertainty": 0.0,
            "diffusion_time": 0.0,
        },
    },
    "features": {
        "scaler": "auto",
        "pca_dim_threshold": 128,
        "pca_max_components": 128,
        "pca_variance": 0.95,
        "clip_outliers": True,
        "clip_sigma": 5.0,
    },
}

# Semi CV/NLP: FTP + fixed T=50 + light TAPS (no DANC collapse on high-d)
SEMI_CVNLP: Dict[str, Any] = {
    "train": {
        "contrastive": True,
        "contrastive_alpha": 0.06,
        "contrastive_alpha_unsupervised": 0.05,
        "hard_negative_mining": True,
        "contrastive_gamma": 1.0,
        "contrastive_adaptive_alpha": True,
    },
    "diffusion": {
        "num_timesteps": 50,
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "scheduler": "linear",
        "time_emb_dim": 4,
        "time_emb_type": "sinusoidal",
    },
    "adadae": {
        "use_danc": False,
        "use_scs": False,
        "use_ftp": True,
        "use_multiview": False,
        "use_uncertainty_view": False,
        "use_dte_view": False,
        "use_rejection_training": False,
        "contrastive_pairing": "taps",
        "fusion_mode": "fixed",
        "fusion_weights": {
            "reconstruction": 1.0,
            "latent": 0.0,
            "residual": 0.0,
            "uncertainty": 0.0,
            "diffusion_time": 0.0,
        },
    },
    "features": {
        "scaler": "auto",
        "pca_dim_threshold": 128,
        "pca_max_components": 128,
        "pca_variance": 0.95,
        "clip_outliers": True,
        "clip_sigma": 5.0,
    },
}

POLICY_NAMES = {
    "baseline_ddae": "baseline_ddae",
    "unsup_ssts": "unsup_ssts",
    "semi_cvnlp": "semi_cvnlp",
}


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def resolve_policy_name(setting: str, category: str) -> str:
    """Return routed policy id for (setting, dataset family)."""
    if setting == "unsupervised":
        return POLICY_NAMES["unsup_ssts"]
    if setting == "semi-supervised" and category in ("cv", "nlp"):
        return POLICY_NAMES["semi_cvnlp"]
    return POLICY_NAMES["baseline_ddae"]


def policy_overrides(policy_name: str) -> Dict[str, Any]:
    if policy_name == POLICY_NAMES["unsup_ssts"]:
        return UNSUP_SSTS
    if policy_name == POLICY_NAMES["semi_cvnlp"]:
        return SEMI_CVNLP
    return BASELINE_DDAE


def apply_routed_config(
    config: Dict[str, Any],
    setting: str,
    category: str = "classical",
) -> Dict[str, Any]:
    """Merge routed policy overrides into a copy of config."""
    adadae_cfg = config.get("adadae", {})
    if str(adadae_cfg.get("policy", "static")) != "routed":
        return config
    name = resolve_policy_name(setting, category)
    overrides = policy_overrides(name)
    out = _deep_update(config, overrides)
    out.setdefault("adadae", {})["resolved_policy"] = name
    return out
