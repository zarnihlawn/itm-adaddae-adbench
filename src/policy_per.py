"""AdaDDAE-PER: single-pass frozen stack of v2→v5.1 hybrid rules + beat-paper freezes.

Combines:
  - v4.1 routed exceptions + meta routing_rules (DAMP off)
  - v5.1 MCE modality targets
  - v5.1 SMC semi-tail fusion
  - GATE v2 selective ensemble
  - selective A6 (orbit/locus/spiral/helix/…)
  - protect_baseline_semi + feature/train overrides
into one ``policy: per`` resolve step (no post-hoc guarded merge).
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .config import PROJECT_ROOT
from .policy import (
    SEMI_SMC_TAIL,
    _deep_update,
    load_policy_exceptions,
    policy_overrides,
    resolve_policy_name,
)

_UPGRADES_CACHE: Optional[Dict[str, Any]] = None
_UPGRADES_CACHE_PATH: Optional[str] = None


def load_per_upgrades(path: Optional[str | Path] = None) -> Dict[str, Any]:
    global _UPGRADES_CACHE, _UPGRADES_CACHE_PATH
    if path is None:
        path = PROJECT_ROOT / "configs" / "adadae_per_upgrades.yaml"
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    key = str(p.resolve())
    if _UPGRADES_CACHE is not None and _UPGRADES_CACHE_PATH == key:
        return _UPGRADES_CACHE
    if not p.exists():
        data: Dict[str, Any] = {}
    else:
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    _UPGRADES_CACHE = data
    _UPGRADES_CACHE_PATH = key
    return data


def clear_per_upgrades_cache() -> None:
    global _UPGRADES_CACHE, _UPGRADES_CACHE_PATH
    _UPGRADES_CACHE = None
    _UPGRADES_CACHE_PATH = None


def _mce_modality(dataset_name: str, upgrades: Dict[str, Any]) -> Optional[str]:
    mce = upgrades.get("mce") or {}
    for modality in ("cv", "nlp", "classical"):
        if dataset_name in (mce.get(modality) or []):
            return modality
    return None


def resolve_per_policy_name(
    setting: str,
    category: str = "classical",
    dataset_name: str = "",
    meta: Optional[Dict[str, float]] = None,
    exceptions: Optional[Dict[str, Any]] = None,
) -> str:
    """Base routed policy id under PER exceptions (DAMP off)."""
    exc = exceptions or load_policy_exceptions(
        PROJECT_ROOT / "configs" / "adadae_per_exceptions.yaml"
    )
    if "use_damp" not in exc:
        exc = dict(exc)
        exc["use_damp"] = False
    return resolve_policy_name(
        setting=setting,
        category=category,
        dataset_name=dataset_name,
        meta=meta,
        exceptions=exc,
    )


def apply_per_upgrades(
    config: Dict[str, Any],
    setting: str,
    category: str,
    dataset_name: str,
    base_policy: str,
    upgrades: Optional[Dict[str, Any]] = None,
    exceptions: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Layer MCE / SMC / GATE / selective A6 + feature/train overrides."""
    upgrades = upgrades or load_per_upgrades()
    out = copy.deepcopy(config)
    adadae = out.setdefault("adadae", {})
    tags: list[str] = [base_policy]

    protected = set(upgrades.get("protect_baseline_semi") or [])
    is_protected = setting == "semi-supervised" and dataset_name in protected

    modality = _mce_modality(dataset_name, upgrades)
    block_semi_nlp = bool((upgrades.get("mce") or {}).get("block_semi_nlp", True))
    mce_ok = modality is not None and not is_protected
    if mce_ok and block_semi_nlp and setting == "semi-supervised" and (
        category == "nlp" or modality == "nlp"
    ):
        mce_ok = False
    if mce_ok:
        adadae["use_mce"] = True
        adadae["mce_modality"] = modality
        adadae["mce_block_semi_nlp"] = True
        tags.append(f"mce_{modality}")

    smc_semi = (upgrades.get("smc") or {}).get("semi") or []
    if (
        not is_protected
        and setting == "semi-supervised"
        and dataset_name in smc_semi
    ):
        out = _deep_update(out, SEMI_SMC_TAIL)
        if mce_ok:
            out.setdefault("adadae", {})["use_mce"] = True
            out["adadae"]["mce_modality"] = modality
            out["adadae"]["mce_block_semi_nlp"] = True
        tags.append("smc")

    gate_cfg = upgrades.get("gate") or {}
    gate_list = gate_cfg.get(setting) or []
    if not is_protected and dataset_name in gate_list:
        out.setdefault("adadae", {})["use_gate"] = True
        tags.append("gate")

    a6 = upgrades.get("a6") or {}
    a6_flag_map = {
        "nautilus": "use_nautilus",
        "apex": "use_apex",
        "orbit": "use_orbit",
        "ridge": "use_ridge",
        "delta": "use_delta",
        "locus": "use_locus",
        "spiral": "use_spiral",
        "helix": "use_helix",
    }
    if not is_protected:
        for mod, flag in a6_flag_map.items():
            ds_list = (a6.get(mod) or {}).get(setting) or []
            if dataset_name in ds_list:
                out.setdefault("adadae", {})[flag] = True
                tags.append(mod)
    else:
        # Hard strip overlays on protected classical
        for flag in a6_flag_map.values():
            out.setdefault("adadae", {})[flag] = False
        out["adadae"]["use_mce"] = False
        out["adadae"]["use_gate"] = False
        tags = [base_policy, "protect"]

    # Feature / train overrides from exceptions
    exc = exceptions or {}
    feat_ov = (exc.get("feature_overrides") or {}).get(dataset_name)
    if isinstance(feat_ov, dict):
        out.setdefault("features", {})
        out["features"] = _deep_update(out.get("features") or {}, feat_ov)
        tags.append("ftp_ov")
    train_ov = (exc.get("train_overrides") or {}).get(dataset_name)
    if isinstance(train_ov, dict):
        out.setdefault("train", {})
        out["train"] = _deep_update(out.get("train") or {}, train_ov)
        tags.append("train_ov")

    # Baseline / champion-style recipes must full-sum score like AnoDDAE
    if base_policy in (
        "baseline_ddae",
        "semi_nlp_baseline",
        "semi_nlp_frozen",
        "champion_semi",
    ):
        out.setdefault("adadae", {})["use_scs"] = False
        out["adadae"]["scs_mode"] = "full_sum"
        out["adadae"]["scs_full_sum_ablation"] = True

    out.setdefault("adadae", {})["policy"] = "per"
    out["adadae"]["resolved_policy"] = "per:" + "+".join(tags)
    return out


def apply_per_config(
    config: Dict[str, Any],
    setting: str,
    category: str = "classical",
    dataset_name: str = "",
    meta: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Full PER resolve: routed base + upgrades + overrides."""
    adadae_cfg = config.get("adadae", {})
    exc_path = adadae_cfg.get("exceptions_file") or "configs/adadae_per_exceptions.yaml"
    exceptions = load_policy_exceptions(exc_path)
    exceptions = dict(exceptions)
    exceptions["use_damp"] = False

    base_name = resolve_policy_name(
        setting=setting,
        category=category,
        dataset_name=dataset_name,
        meta=meta,
        exceptions=exceptions,
    )
    overrides = policy_overrides(base_name)
    out = _deep_update(config, overrides)

    upgrades_path = adadae_cfg.get("upgrades_file") or "configs/adadae_per_upgrades.yaml"
    upgrades = load_per_upgrades(upgrades_path)
    return apply_per_upgrades(
        out,
        setting=setting,
        category=category,
        dataset_name=dataset_name,
        base_policy=base_name,
        upgrades=upgrades,
        exceptions=exceptions,
    )
