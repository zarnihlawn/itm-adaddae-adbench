#!/usr/bin/env python3
"""Print AdaDDAE-PER resolved policy for all ADBench datasets × settings."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_yaml
from src.data.datasets import build_registry
from src.policy import apply_routed_config


def main() -> int:
    cfg = load_yaml(PROJECT_ROOT / "configs" / "adadae_per.yaml")
    adbench = PROJECT_ROOT / cfg["paths"]["adbench_root"]
    if not adbench.is_dir():
        # still dump from classical file list with category classical/cv/nlp stubs
        print(f"NOTE: ADBench root missing at {adbench} — using registry paths only")
    specs = build_registry(adbench)
    print(f"{'dataset':20s} {'cat':8s} {'setting':18s} resolved_policy")
    print("-" * 90)
    for spec in specs:
        for setting in ("unsupervised", "semi-supervised"):
            out = apply_routed_config(
                cfg,
                setting=setting,
                category=spec.category,
                dataset_name=spec.name,
                meta={"n": 5000.0, "d": 32.0},
            )
            pol = out.get("adadae", {}).get("resolved_policy", "?")
            flags = []
            a = out.get("adadae", {})
            if a.get("use_mce"):
                flags.append(f"mce={a.get('mce_modality')}")
            if a.get("fusion_mode") == "smc":
                flags.append("smc")
            if a.get("use_gate"):
                flags.append("gate")
            extra = (" [" + ",".join(flags) + "]") if flags else ""
            print(f"{spec.name:20s} {spec.category:8s} {setting:18s} {pol}{extra}")
    print(f"\nTotal specs: {len(specs)} (expect 57)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
