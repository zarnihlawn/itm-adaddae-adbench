"""AdaDDAE-PER: single-pass frozen stack of v2→v5.1 hybrid rules.

Combines:
  - v4.1 routed exceptions + meta routing_rules (DAMP off)
  - v5.1 MCE modality targets
  - v5.1 SMC semi-tail fusion
  - GATE v2 selective ensemble
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


def load_per_upgrades(path: Optional[str | Path] = None) -> Dict[str, Any]:
    global _UPGRADES_CACHE
    if path is None and _UPGRADES_CACHE is not None:
        return _UPGRADES_CACHE
    if path is None:
        path = PROJECT_ROOT / "configs" / "adadae_per_upgrades.yaml"
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        data: Dict[str, Any] = {}
    else:
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    if path is None or p == PROJECT_ROOT / "configs" / "adadae_per_upgrades.yaml":
        _UPGRADES_CACHE = data
    return data


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
) -> Dict[str, Any]:
    """Layer v5.1 MCE / SMC / GATE / selective A6 flags onto a routed config copy."""
    upgrades = upgrades or load_per_upgrades()
    out = copy.deepcopy(config)
    adadae = out.setdefault("adadae", {})
    tags: list[str] = [base_policy]

    modality = _mce_modality(dataset_name, upgrades)
    block_semi_nlp = bool((upgrades.get("mce") or {}).get("block_semi_nlp", True))
    mce_ok = modality is not None
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
    if setting == "semi-supervised" and dataset_name in smc_semi:
        out = _deep_update(out, SEMI_SMC_TAIL)
        # keep MCE flags if already set
        if mce_ok:
            out.setdefault("adadae", {})["use_mce"] = True
            out["adadae"]["mce_modality"] = modality
            out["adadae"]["mce_block_semi_nlp"] = True
        tags.append("smc")

    gate_cfg = upgrades.get("gate") or {}
    gate_list = gate_cfg.get(setting) or []
    if dataset_name in gate_list:
        out.setdefault("adadae", {})["use_gate"] = True
        tags.append("gate")

    # Loop 7: selective A6 (nautilus / apex / orbit / ridge / delta)
    a6 = upgrades.get("a6") or {}
    a6_flag_map = {
        "nautilus": "use_nautilus",
        "apex": "use_apex",
        "orbit": "use_orbit",
        "ridge": "use_ridge",
        "delta": "use_delta",
    }
    for mod, flag in a6_flag_map.items():
        ds_list = (a6.get(mod) or {}).get(setting) or []
        if dataset_name in ds_list:
            out.setdefault("adadae", {})[flag] = True
            tags.append(mod)

    # Loop 5: baseline / champion-style recipes must full-sum score like AnoDDAE
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
    """Full PER resolve: routed base + v5.1 upgrades."""
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
    )
