#!/usr/bin/env python3
"""Build AdaDDAE v5 hybrid from v4.1 base + v5 patches."""
from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Build v5 hybrid")
    parser.add_argument("--base", default="results/adadae_v41_hybrid/metrics/completed.json")
    parser.add_argument("--phase1-patch", default="results/adadae_v5_phase1/metrics/completed.json")
    parser.add_argument("--mce-patch", default="results/adadae_v5_mce/metrics/completed.json")
    parser.add_argument("--smc-patch", default="results/adadae_v5_smc/metrics/completed.json")
    parser.add_argument("--gate-patch", default="results/adadae_v5_gate/metrics/completed.json")
    parser.add_argument("--min-v41-unsup", type=float, default=37.83)
    parser.add_argument("--min-v41-semi", type=float, default=62.01)
    parser.add_argument("--out", default="results/adadae_v5_hybrid/metrics/completed.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    base_path = Path(args.base)
    if not base_path.is_absolute():
        base_path = PROJECT_ROOT / base_path
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path

    merge_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "merge_completed.py"),
        "--semi-classical",
        str(base_path),
        "--semi-cvnlp",
        str(base_path),
        "--semi-cvnlp-source",
        "path",
        "--unsup",
        str(base_path),
        "--out",
        str(out_path),
    ]

    for patch_arg, patch_path in [
        ("--patch", args.phase1_patch),
        ("--patch2", args.mce_patch),
        ("--patch3", args.smc_patch),
        ("--patch4", args.gate_patch),
    ]:
        p = Path(patch_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.exists():
            merge_cmd.extend([patch_arg, str(p)])

    subprocess.run(merge_cmd, cwd=PROJECT_ROOT, check=True)

    completed = load_state(out_path).get("completed", {})
    unsup_mean = mean_pr(completed, "unsupervised")
    semi_mean = mean_pr(completed, "semi-supervised")
    combined = (unsup_mean + semi_mean) / 2.0
    print(f"\nUnsup mean PR: {unsup_mean:.2f}% (v41 min {args.min_v41_unsup}%)")
    print(f"Semi mean PR:  {semi_mean:.2f}% (v41 min {args.min_v41_semi}%)")
    print(f"Combined macro PR: {combined:.2f}%")

    gate_ok = semi_mean >= args.min_v41_semi - 1e-6 and unsup_mean >= args.min_v41_unsup - 1e-6
    if not gate_ok and not args.force:
        print("GATE FAIL — use --force to write anyway")
        sys.exit(1)

    print(f"Wrote {out_path}")
    sys.exit(0 if gate_ok else 1)


if __name__ == "__main__":
    main()
