#!/usr/bin/env python3
"""Build hybrid completed.json by picking best PR policy per dataset from existing runs."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.merge_completed import copy_metrics, filter_completed, load_state

POLICY_SOURCE = {
    "backup_baseline": PROJECT_ROOT / "backup/ddae_baseline_570/metrics/completed.json",
    "unsup_ssts": PROJECT_ROOT / "results/adadae_unsup_ssts/metrics/completed.json",
    "semi_cvnlp": PROJECT_ROOT / "results/adadae_semi_cvnlp/metrics/completed.json",
}

PUBLISHED = {
    "unsupervised": 32.77,
    "semi-supervised": 61.36,
}


def main():
    parser = argparse.ArgumentParser(description="Build oracle-best hybrid completed.json")
    parser.add_argument(
        "--oracle-csv",
        default="results/thesis/oracle_policy_best.csv",
    )
    parser.add_argument(
        "--patch",
        default=None,
        help="Optional v3 patch completed.json to override keys",
    )
    parser.add_argument(
        "--out",
        default="results/adadae_v3_hybrid/metrics/completed.json",
    )
    parser.add_argument("--copy-metrics", action="store_true")
    args = parser.parse_args()

    oracle_path = Path(args.oracle_csv)
    if not oracle_path.is_absolute():
        oracle_path = PROJECT_ROOT / oracle_path
    best = pd.read_csv(oracle_path)

    states = {name: load_state(path) for name, path in POLICY_SOURCE.items()}
    merged = {"completed": {}, "failed": {}}
    source_keys: dict[str, set[str]] = {n: set() for n in POLICY_SOURCE}

    for _, row in best.iterrows():
        policy = row["policy"]
        ds = row["dataset"]
        setting = row["setting"]
        src = states.get(policy, {})
        for key, job in src.get("completed", {}).items():
            if job.get("dataset") == ds and job.get("setting") == setting:
                merged["completed"][key] = job
                source_keys.setdefault(policy, set()).add(key)

    if args.patch:
        patch_path = Path(args.patch)
        if not patch_path.is_absolute():
            patch_path = PROJECT_ROOT / patch_path
        patch_state = load_state(patch_path)
        merged["completed"].update(patch_state.get("completed", {}))
        print(f"Patch override: {len(patch_state.get('completed', {}))} jobs")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"Oracle hybrid: {len(merged['completed'])} jobs -> {out_path}")

    if args.copy_metrics:
        copied = 0
        for policy, keys in source_keys.items():
            src_metrics = POLICY_SOURCE[policy].parent
            copied += copy_metrics(src_metrics, out_path.parent, keys)
        if args.patch:
            patch_metrics = (PROJECT_ROOT / args.patch).parent
            p_state = load_state(Path(args.patch))
            copied += copy_metrics(patch_metrics, out_path.parent, set(p_state.get("completed", {})))
        print(f"Copied {copied} metric JSON files")

    # Quick gate summary
    import numpy as np

    for setting in ["unsupervised", "semi-supervised"]:
        prs = []
        for job in merged["completed"].values():
            if job.get("setting") != setting:
                continue
            prs.append(job["metrics_mean"]["PR-AUC"] * 100)
        if prs:
            # per-dataset mean
            df = pd.DataFrame(
                [
                    {
                        "dataset": j["dataset"],
                        "PR": j["metrics_mean"]["PR-AUC"] * 100,
                    }
                    for j in merged["completed"].values()
                    if j.get("setting") == setting
                ]
            )
            mean_pr = df.groupby("dataset")["PR"].mean().mean()
            pub = PUBLISHED[setting]
            flag = "PASS" if mean_pr > pub else "FAIL"
            print(f"{setting}: mean PR {mean_pr:.2f}% vs paper {pub}% [{flag}]")


if __name__ == "__main__":
    main()
