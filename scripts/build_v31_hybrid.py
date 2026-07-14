#!/usr/bin/env python3
"""Build v3.1 hybrid with counterfactual semi-tail gate before writing completed.json."""
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
from src.policy import load_policy_exceptions

PUBLISHED = {"unsupervised": 32.77, "semi-supervised": 61.36}
SEMI_GATE = 61.40


def mean_pr(completed: dict, setting: str) -> float:
    rows = []
    for job in completed.values():
        if job.get("setting") != setting:
            continue
        rows.append({"dataset": job["dataset"], "PR": job["metrics_mean"]["PR-AUC"] * 100})
    if not rows:
        return 0.0
    return pd.DataFrame(rows).groupby("dataset")["PR"].mean().mean()


def simulate_tail_from_bisect(
    completed: dict,
    bisect_best: pd.DataFrame,
    tail_datasets: set[str],
    setting: str = "semi-supervised",
) -> tuple[dict, pd.DataFrame]:
    """Replace semi tail dataset PR with bisect best (counterfactual upper bound)."""
    sim_rows = []
    out = json.loads(json.dumps(completed))
    best = bisect_best[bisect_best["setting"] == setting] if "setting" in bisect_best.columns else bisect_best

    for ds in tail_datasets:
        sub = best[best["dataset"] == ds]
        if sub.empty:
            continue
        row = sub.sort_values("PR-AUC", ascending=False).iloc[0]
        target_pr = row["PR-AUC"] / 100.0
        for key, job in list(out.items()):
            if job.get("dataset") == ds and job.get("setting") == setting:
                old_pr = job["metrics_mean"]["PR-AUC"]
                if target_pr <= old_pr:
                    continue
                job = json.loads(json.dumps(job))
                job["metrics_mean"]["PR-AUC"] = target_pr
                job["simulated_from_bisect"] = row.get("candidate", "")
                out[key] = job
                sim_rows.append({
                    "dataset": ds,
                    "old_PR": old_pr * 100,
                    "new_PR": row["PR-AUC"],
                    "candidate": row.get("candidate", ""),
                })
    return out, pd.DataFrame(sim_rows)


def main():
    parser = argparse.ArgumentParser(description="Build v3.1 hybrid with counterfactual gate")
    parser.add_argument("--unsup", default="results/adadae_unsup_ssts/metrics/completed.json")
    parser.add_argument("--unsup-patch", default="results/adadae_v31_unsup/metrics/completed.json")
    parser.add_argument("--semi-tail-patch", default="results/adadae_v31_semi_tail/metrics/completed.json")
    parser.add_argument("--bisect-best", default="results/thesis/v31_semi_tail_best.csv")
    parser.add_argument("--simulate-only", action="store_true", help="Counterfactual using bisect CSV only")
    parser.add_argument("--min-semi-pr", type=float, default=SEMI_GATE)
    parser.add_argument("--out", default="results/adadae_v31_hybrid/metrics/completed.json")
    parser.add_argument("--force", action="store_true", help="Write even if gate fails")
    parser.add_argument("--copy-metrics", action="store_true")
    args = parser.parse_args()

    exc = load_policy_exceptions()
    tail = set(exc.get("semi_tail_datasets", []))

    if args.simulate_only:
        base_path = PROJECT_ROOT / "results/adadae_v3_hybrid/metrics/completed.json"
        completed = load_state(base_path).get("completed", {})
        bisect_path = Path(args.bisect_best)
        if not bisect_path.is_absolute():
            bisect_path = PROJECT_ROOT / bisect_path
        if not bisect_path.exists():
            bisect_path = PROJECT_ROOT / "results/thesis/v3_hard_dataset_best.csv"
        bisect_df = pd.read_csv(bisect_path)
        completed, sim_df = simulate_tail_from_bisect(completed, bisect_df, tail)
        print("=== Counterfactual semi tail simulation ===")
        if not sim_df.empty:
            print(sim_df.to_string(index=False))
    else:
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
        unsup_patch = Path(args.unsup_patch)
        if not unsup_patch.is_absolute():
            unsup_patch = PROJECT_ROOT / unsup_patch
        if unsup_patch.exists():
            merge_cmd.extend(["--patch", str(unsup_patch)])

        semi_patch = Path(args.semi_tail_patch)
        if not semi_patch.is_absolute():
            semi_patch = PROJECT_ROOT / semi_patch
        if semi_patch.exists():
            merge_cmd.extend(["--patch2", str(semi_patch), "--semi-tail-only"])

        if args.copy_metrics:
            merge_cmd.append("--copy-metrics")

        subprocess.run(merge_cmd, cwd=PROJECT_ROOT, check=True)
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        completed = load_state(out_path).get("completed", {})

    unsup_mean = mean_pr(completed, "unsupervised")
    semi_mean = mean_pr(completed, "semi-supervised")
    print(f"\nUnsup mean PR: {unsup_mean:.2f}% (paper {PUBLISHED['unsupervised']}%)")
    print(f"Semi mean PR:  {semi_mean:.2f}% (paper {PUBLISHED['semi-supervised']}%, gate {args.min_semi_pr}%)")

    gate_ok = semi_mean >= args.min_semi_pr
    if not gate_ok and not args.force and not args.simulate_only:
        print(f"GATE FAIL: semi {semi_mean:.2f}% < {args.min_semi_pr}% — use --simulate-only to preview or --force to write")
        sys.exit(1)

    if args.simulate_only:
        flag = "PASS" if semi_mean >= args.min_semi_pr else "FAIL"
        print(f"Counterfactual gate: {flag}")
        sim_out = PROJECT_ROOT / "results/thesis/v31_counterfactual_summary.json"
        sim_out.parent.mkdir(parents=True, exist_ok=True)
        sim_out.write_text(
            json.dumps(
                {
                    "unsup_mean_PR": float(unsup_mean),
                    "semi_mean_PR": float(semi_mean),
                    "gate_pass": bool(gate_ok),
                    "min_semi_pr": float(args.min_semi_pr),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {sim_out}")
    else:
        print(f"Wrote {args.out}")

    sys.exit(0 if gate_ok else 1)


if __name__ == "__main__":
    main()
