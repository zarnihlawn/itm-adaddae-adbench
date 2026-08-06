#!/usr/bin/env python3
"""Dump adaptive PER recipe map for all 57 ADBench protocol datasets.

Resolves live policy (exceptions + upgrades) and records NPZ paths from the
registry. Run locally before spending Vast GPU:

  python scripts/dump_adaptive_recipe_map.py
  # → results/adadae_per/thesis/adaptive_recipe_map_57.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.apply_hard_tail_freeze import (  # noqa: E402
    PHASE0_LOCKS,
    PHASE0_STRIP_A6_SEMI,
    PHASE0_STRIP_METHOD_LIFTS,
)
from src.config import load_config, load_yaml  # noqa: E402
from src.data.datasets import build_registry  # noqa: E402
from src.policy_per import apply_per_config, clear_per_upgrades_cache  # noqa: E402


def main() -> int:
    clear_per_upgrades_cache()
    cfg = load_config(str(PROJECT_ROOT / "configs" / "adadae_per.yaml"), hardware="cpu")
    upgrades = load_yaml(PROJECT_ROOT / "configs" / "adadae_per_upgrades.yaml")
    adbench = Path(cfg["paths"]["adbench_root"])
    if not adbench.is_absolute():
        adbench = (PROJECT_ROOT / adbench).resolve()
    registry = build_registry(adbench)

    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    for entry in registry:
        ds = entry.name
        cat = entry.category
        out = apply_per_config(
            cfg, "semi-supervised", cat, ds, meta={"n": 1000.0, "d": 20.0}
        )
        ad = out["adadae"]
        pol = str(ad.get("resolved_policy") or "")
        row = {
            "dataset": ds,
            "category": cat,
            "n_npz": len(entry.relative_paths),
            "npz_paths": list(entry.relative_paths),
            "resolved_policy": pol,
            "use_rejection_training": bool(ad.get("use_rejection_training")),
            "use_apex": bool(ad.get("use_apex")),
            "use_delta": bool(ad.get("use_delta")),
            "use_helix": bool(ad.get("use_helix")),
            "use_ridge": bool(ad.get("use_ridge")),
            "contrastive": bool((out.get("train") or {}).get("contrastive")),
            "cal_fuse": ad.get("fusion_mode") == "calibrated",
            "protected": "protect" in pol,
            "phase0_lock": ds in PHASE0_LOCKS,
            "train_overrides": (out.get("train") or {}),
        }
        # Integrity checks for last-shot map
        if ds in PHASE0_LOCKS:
            if "baseline_ddae" not in pol or ad.get("use_rejection_training"):
                errors.append(f"{ds}: PHASE0 lock broken pol={pol}")
        if ds in ("backdoor", "thyroid", "optdigits", "Amazon", "Yelp", "20newsgroups", "Agnews"):
            if (out.get("train") or {}).get("contrastive"):
                errors.append(f"{ds}: taps still on")
        if ds in ("backdoor", "optdigits", "thyroid", "vertebral"):
            if ad.get("fusion_mode") == "calibrated":
                errors.append(f"{ds}: cal_fuse still on")
        if ds == "fraud" and (ad.get("use_apex") or ad.get("use_delta")):
            errors.append(f"{ds}: apex/delta still on")
        if ds == "WPBC" and ad.get("use_helix"):
            errors.append(f"{ds}: helix still on")
        rows.append(row)

    lifts = upgrades.get("method_lifts") or {}
    for lift_key, ds_names in PHASE0_STRIP_METHOD_LIFTS.items():
        lst = set(lifts.get(lift_key) or [])
        leaked = [n for n in ds_names if n in lst]
        if leaked:
            errors.append(f"lift {lift_key} still lists {leaked}")

    a6 = upgrades.get("a6") or {}
    for mod, ds_names in PHASE0_STRIP_A6_SEMI.items():
        semi = set((a6.get(mod) or {}).get("semi-supervised") or [])
        leaked = [n for n in ds_names if n in semi]
        if leaked:
            errors.append(f"a6.{mod} still lists {leaked}")

    report = {
        "title": "AdaDDAE-PER adaptive recipe map (57 ADBench)",
        "version": upgrades.get("version"),
        "n_datasets": len(rows),
        "adbench_root": str(adbench),
        "pass": not errors and len(rows) == 57,
        "errors": errors,
        "phase0_locks": dict(PHASE0_LOCKS),
        "phase0_strip_method_lifts": {
            k: list(v) for k, v in PHASE0_STRIP_METHOD_LIFTS.items()
        },
        "phase0_strip_a6_semi": {
            k: list(v) for k, v in PHASE0_STRIP_A6_SEMI.items()
        },
        "datasets": rows,
    }
    out_path = PROJECT_ROOT / "results/adadae_per/thesis/adaptive_recipe_map_57.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "n": len(rows), "errors": errors, "out": str(out_path)}, indent=2))
    if len(rows) != 57:
        print(f"WARN: expected 57 datasets, got {len(rows)}", file=sys.stderr)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
