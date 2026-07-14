"""Setting-, family-, and dataset-aware AdaDDAE v3 policy routing."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .config import PROJECT_ROOT

# DDAE-faithful baseline (semi classical + fallbacks)
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

# Semi CV: FTP + fixed T=50, contrastive off
SEMI_CVNLP_FTP: Dict[str, Any] = {
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
        "use_ftp": True,
        "use_multiview": False,
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

# Semi CV/NLP: FTP + fixed T=50 + light TAPS (v2 — use only when bisect proves win)
SEMI_CVNLP_TAPS: Dict[str, Any] = {
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

# Speech / hard classical specialist
SEMI_SPEECH_SPECIALIST: Dict[str, Any] = {
    "train": {
        "contrastive": False,
        "contrastive_alpha": 0.0,
        "hard_negative_mining": False,
    },
    "diffusion": {
        "num_timesteps": 80,
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
        "use_rejection_training": True,
        "rejection_quantile": 0.95,
        "rejection_min_weight": 0.1,
        "rejection_warmup_epochs": 1,
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
        "scaler": "robust",
        "pca_dim_threshold": 99999,
        "clip_outliers": False,
    },
}

POLICY_REGISTRY: Dict[str, Dict[str, Any]] = {
    "baseline_ddae": BASELINE_DDAE,
    "unsup_ssts": UNSUP_SSTS,
    "unsup_baseline_fallback": BASELINE_DDAE,
    "semi_cvnlp_ftp": SEMI_CVNLP_FTP,
    "semi_cvnlp_taps_light": SEMI_CVNLP_TAPS,
    "semi_nlp_baseline": BASELINE_DDAE,
    "semi_speech_specialist": SEMI_SPEECH_SPECIALIST,
    # Legacy v2 alias
    "semi_cvnlp": SEMI_CVNLP_TAPS,
}

DEFAULT_EXCEPTIONS: Dict[str, Any] = {
    "unsup_baseline_fallback": ["vowels", "letter", "skin", "fault", "wine", "glass"],
    "semi_nlp_baseline": ["Agnews", "20newsgroups", "Amazon", "Imdb", "Yelp"],
    "semi_specialists": {"speech": "semi_speech_specialist"},
    "semi_cv_policy": "semi_cvnlp_ftp",
    "unsup_nlp_baseline": ["Agnews", "Amazon", "Imdb", "Yelp", "20newsgroups"],
}


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_policy_exceptions(path: Optional[str | Path] = None) -> Dict[str, Any]:
    if path is None:
        path = PROJECT_ROOT / "configs" / "policy_exceptions.yaml"
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        return copy.deepcopy(DEFAULT_EXCEPTIONS)
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    merged = copy.deepcopy(DEFAULT_EXCEPTIONS)
    merged.update(data)
    return merged


def resolve_policy_name(
    setting: str,
    category: str = "classical",
    dataset_name: str = "",
    meta: Optional[Dict[str, float]] = None,
    exceptions: Optional[Dict[str, Any]] = None,
) -> str:
    """Return routed policy id for (setting, family, dataset, meta)."""
    exc = exceptions or load_policy_exceptions()
    meta = meta or {}
    n = int(meta.get("n", 0))
    d = int(meta.get("d", 0))

    if setting == "unsupervised":
        if dataset_name in exc.get("unsup_baseline_fallback", []):
            return "unsup_baseline_fallback"
        if dataset_name in exc.get("unsup_nlp_baseline", []) and category == "nlp":
            return "unsup_baseline_fallback"
        # Small-n high-d classical: SSTS can collapse (vowels, letter evidence)
        if category == "classical" and n > 0 and d > 0 and n < 2000 and d > 10:
            if dataset_name in {"vowels", "letter"}:
                return "unsup_baseline_fallback"
        return "unsup_ssts"

    if setting == "semi-supervised":
        specialists = exc.get("semi_specialists", {})
        if dataset_name in specialists:
            return str(specialists[dataset_name])

        if dataset_name in exc.get("semi_nlp_baseline", []) and category == "nlp":
            return "semi_nlp_baseline"

        if category == "cv":
            return str(exc.get("semi_cv_policy", "semi_cvnlp_ftp"))

        if category == "nlp":
            return "semi_nlp_baseline"

        return "baseline_ddae"

    return "baseline_ddae"


def policy_overrides(policy_name: str) -> Dict[str, Any]:
    if policy_name not in POLICY_REGISTRY:
        return BASELINE_DDAE
    return POLICY_REGISTRY[policy_name]


def apply_routed_config(
    config: Dict[str, Any],
    setting: str,
    category: str = "classical",
    dataset_name: str = "",
    meta: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Merge routed policy overrides into a copy of config."""
    adadae_cfg = config.get("adadae", {})
    if str(adadae_cfg.get("policy", "static")) != "routed":
        return config

    exc_path = adadae_cfg.get("exceptions_file")
    exceptions = load_policy_exceptions(exc_path) if exc_path else load_policy_exceptions()
    name = resolve_policy_name(
        setting=setting,
        category=category,
        dataset_name=dataset_name,
        meta=meta,
        exceptions=exceptions,
    )
    overrides = policy_overrides(name)
    out = _deep_update(config, overrides)
    out.setdefault("adadae", {})["resolved_policy"] = name
    return out
