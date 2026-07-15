#!/usr/bin/env python3
"""Build AdaDDAE v5.1 hybrid with regression-guarded merge."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.merge_v5_guarded import merge_layers_guarded


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
    parser = argparse.ArgumentParser(description="Build v5.1 guarded hybrid")
    parser.add_argument("--base", default="results/adadae_v41_hybrid/metrics/completed.json")
    parser.add_argument("--phase1-patch", default="results/adadae_v51_phase1/metrics/completed.json")
    parser.add_argument("--mce-patch", default="results/adadae_v51_mce/metrics/completed.json")
    parser.add_argument("--smc-patch", default="results/adadae_v51_smc/metrics/completed.json")
    parser.add_argument("--gate-patch", default="")
    parser.add_argument("--use-gate", action="store_true")
    parser.add_argument("--guard-epsilon", type=float, default=0.1)
    parser.add_argument("--min-v41-unsup", type=float, default=37.83)
    parser.add_argument("--min-v41-semi", type=float, default=62.01)
    parser.add_argument("--strict-beat-v41", action="store_true")
    parser.add_argument("--out", default="results/adadae_v51_hybrid/metrics/completed.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    base_path = PROJECT_ROOT / args.base
    patches: list[tuple[str, Path]] = [
        ("phase1", PROJECT_ROOT / args.phase1_patch),
        ("mce", PROJECT_ROOT / args.mce_patch),
        ("smc", PROJECT_ROOT / args.smc_patch),
    ]
    gate_path = PROJECT_ROOT / args.gate_patch if args.gate_patch else PROJECT_ROOT / "results/adadae_v51_gate/metrics/completed.json"
    if args.use_gate and gate_path.exists():
        patches.append(("gate", gate_path))

    merged, audit = merge_layers_guarded(base_path, patches, guard_epsilon=args.guard_epsilon)

    out_path = PROJECT_ROOT / args.out
    audit_path = out_path.parent.parent / "thesis" / "merge_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    accepted = sum(1 for a in audit if a["action"] in ("accept", "accept_new"))
    rejected = sum(1 for a in audit if a["action"] == "reject")
    bad_accepts = [a for a in audit if a["action"] == "accept" and a.get("delta_pp", 0) < -0.5]
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps({
            "guard_epsilon_pp": args.guard_epsilon,
            "n_accepted": accepted,
            "n_rejected": rejected,
            "n_bad_accepts_lt_0_5pp": len(bad_accepts),
            "bad_accepts": bad_accepts,
            "entries": audit,
        }, indent=2),
        encoding="utf-8",
    )

    completed = merged["completed"]
    unsup_mean = mean_pr(completed, "unsupervised")
    semi_mean = mean_pr(completed, "semi-supervised")
    combined = (unsup_mean + semi_mean) / 2.0
    print(f"\nUnsup mean PR: {unsup_mean:.2f}% (v41 min {args.min_v41_unsup}%)")
    print(f"Semi mean PR:  {semi_mean:.2f}% (v41 min {args.min_v41_semi}%)")
    print(f"Combined macro PR: {combined:.2f}%")
    print(f"Guarded: {accepted} accepted, {rejected} rejected")

    if args.strict_beat_v41:
        gate_ok = unsup_mean > args.min_v41_unsup and semi_mean > args.min_v41_semi
    else:
        gate_ok = unsup_mean >= args.min_v41_unsup - 1e-6 and semi_mean >= args.min_v41_semi - 1e-6

    if not gate_ok and not args.force:
        print("GATE FAIL — use --force to write anyway")
        sys.exit(1)

    print(f"Wrote {out_path}")
    print(f"Audit {audit_path}")
    sys.exit(0 if gate_ok else 1)


if __name__ == "__main__":
    main()
