#!/usr/bin/env python3
"""Audit DDAE reproduction protocol vs baselines_ddae.yaml / AnoDDAE expectations."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config

EXPECTED = {
    "train.epochs": 100,
    "train.lr": 0.001,
    "train.eval_every": 10,
    "train.early_stop_patience": 30,
    "train.contrastive": False,
    "diffusion.num_timesteps": 50,
    "diffusion.scheduler": "linear",
    "diffusion.beta_end": 0.02,
    "adadae.use_danc": False,
    "adadae.use_scs": False,
    "adadae.use_ftp": False,
    "features.scaler": "standard",
    "features.clip_outliers": False,
    "seeds": [111, 222, 333, 444, 555],
}

PUBLISHED = {
    "semi-supervised": {"PR-AUC": 61.36, "ROC-AUC": 83.17},
    "unsupervised": {"PR-AUC": 32.77, "ROC-AUC": 74.08},
}


def get_nested(cfg: dict, key: str):
    parts = key.split(".")
    cur = cfg
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def main():
    cfg_path = PROJECT_ROOT / "configs" / "baselines_ddae.yaml"
    cfg = load_config(cfg_path)

    report = {"config": str(cfg_path), "checks": [], "baseline_vs_paper": {}}
    ok = True
    for key, expected in EXPECTED.items():
        actual = get_nested(cfg, key) if "." in key else cfg.get(key.split(".")[0])
        if key == "seeds":
            actual = cfg.get("seeds")
        else:
            actual = get_nested(cfg, key)
        passed = actual == expected
        ok = ok and passed
        report["checks"].append({"key": key, "expected": expected, "actual": actual, "ok": passed})

    backup_compare = PROJECT_ROOT / "backup/ddae_baseline_570/thesis/compare_to_ddae.csv"
    if backup_compare.exists():
        import pandas as pd

        df = pd.read_csv(backup_compare)
        for _, row in df.iterrows():
            setting = row["setting"]
            pub = PUBLISHED.get(setting, {})
            report["baseline_vs_paper"][setting] = {
                "our_PR": float(row["AdaDDAE_PR_AUC"]),
                "paper_PR": pub.get("PR-AUC"),
                "delta_PR": float(row["delta_PR"]),
                "our_ROC": float(row["AdaDDAE_ROC_AUC"]),
                "paper_ROC": pub.get("ROC-AUC"),
                "delta_ROC": float(row["delta_ROC"]),
            }

    report["all_checks_pass"] = ok
    report["repro_gap_semi_PR"] = report.get("baseline_vs_paper", {}).get(
        "semi-supervised", {}
    ).get("delta_PR")
    report["recommendation"] = (
        "Protocol matches DDAE repro config. Semi PR gap vs paper is likely "
        "dataset/method variance, not a config bug. Focus on v3 routing + specialists."
        if ok
        else "Fix protocol mismatches before re-running semi classical jobs."
    )

    out_dir = PROJECT_ROOT / "results" / "thesis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "repro_audit.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Protocol audit: {'PASS' if ok else 'FAIL'}")
    for c in report["checks"]:
        if not c["ok"]:
            print(f"  MISMATCH {c['key']}: expected {c['expected']}, got {c['actual']}")
    if report["baseline_vs_paper"]:
        semi = report["baseline_vs_paper"].get("semi-supervised", {})
        print(
            f"Baseline semi PR {semi.get('our_PR', '?'):.2f}% "
            f"vs paper {semi.get('paper_PR', '?')}% "
            f"(delta {semi.get('delta_PR', '?'):+.2f})"
        )
    print(f"Wrote {out_path}")
    print(report["recommendation"])


if __name__ == "__main__":
    main()
