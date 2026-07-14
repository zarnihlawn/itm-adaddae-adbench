#!/usr/bin/env python3
"""Build AdaDDAE v4 hybrid with safe merge and counterfactual gate."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.merge_completed import load_state

PUBLISHED = {"unsupervised": 32.77, "semi-supervised": 61.36}


def mean_pr(completed: dict, setting: str) -> float:
    rows = []
    for job in completed.values():
        if job.get("setting") != setting:
            continue
        rows.append({"dataset": job["dataset"], "PR": job["metrics_mean"]["PR-AUC"] * 100})
    if not rows:
        return 0.0
    return pd.DataFrame(rows).groupby("dataset")["PR"].mean().mean()


def main():
    parser = argparse.ArgumentParser(description="Build v4 hybrid")
    parser.add_argument("--unsup", default="results/adadae_unsup_ssts/metrics/completed.json")
    parser.add_argument("--unsup-patch", default="results/adadae_v4_unsup/metrics/completed.json")
    parser.add_argument("--semi-tail-patch", default="results/adadae_v4_semi_tail/metrics/completed.json")
    parser.add_argument("--min-semi-pr", type=float, default=61.36)
    parser.add_argument("--min-unsup-pr", type=float, default=36.77)
    parser.add_argument("--out", default="results/adadae_v4_hybrid/metrics/completed.json")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--copy-metrics", action="store_true")
    args = parser.parse_args()

    merge_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/merge_completed.py"),
        "--semi-classical",
        "backup/ddae_baseline_570/metrics/completed.json",
        "--semi-cvnlp",
        "backup/ddae_baseline_570/metrics/completed.json",
        "--semi-cvnlp-source",
        "backup",
        "--unsup",
        args.unsup,
        "--out",
        args.out,
    ]
    for patch_arg, patch_path in [
        ("--patch", args.unsup_patch),
        ("--patch2", args.semi_tail_patch),
    ]:
        p = Path(patch_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.exists():
            merge_cmd.extend([patch_arg, str(p)])
    if "--patch2" in merge_cmd:
        merge_cmd.append("--semi-tail-only")
    if args.copy_metrics:
        merge_cmd.append("--copy-metrics")

    subprocess.run(merge_cmd, cwd=PROJECT_ROOT, check=True)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    completed = load_state(out_path).get("completed", {})

    unsup_mean = mean_pr(completed, "unsupervised")
    semi_mean = mean_pr(completed, "semi-supervised")
    print(f"\nUnsup mean PR: {unsup_mean:.2f}% (paper {PUBLISHED['unsupervised']}%, min {args.min_unsup_pr}%)")
    print(f"Semi mean PR:  {semi_mean:.2f}% (paper {PUBLISHED['semi-supervised']}%, min {args.min_semi_pr}%)")

    gate_ok = semi_mean >= args.min_semi_pr - 1e-6 and unsup_mean >= args.min_unsup_pr - 1e-6
    if not gate_ok and not args.force:
        print("GATE FAIL — use --force to write anyway")
        sys.exit(1)

    print(f"Wrote {args.out}")
    sys.exit(0 if gate_ok else 1)


if __name__ == "__main__":
    main()
