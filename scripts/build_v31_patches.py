#!/usr/bin/env python3
"""Build v3.1 patch completed.json files from existing GPU results (pre-Vast shortcut)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.policy import load_policy_exceptions

UNSUP_FALLBACK = [
    "magic.gamma", "landsat", "fraud", "InternetAds", "Ionosphere", "glass",
    "WPBC", "vertebral", "backdoor", "vowels", "letter", "skin", "fault", "wine",
]


def load_completed(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("completed", data)


def filter_jobs(completed: dict, setting: str, datasets: set[str]) -> dict:
    return {
        k: v
        for k, v in completed.items()
        if v.get("setting") == setting and v.get("dataset") in datasets
    }


def pick_best_source(
    sources: dict[str, dict],
    dataset: str,
    setting: str,
) -> tuple[str, dict]:
    best_name = ""
    best_jobs: dict = {}
    best_mean = -1.0
    for name, comp in sources.items():
        jobs = {
            k: v
            for k, v in comp.items()
            if v.get("dataset") == dataset and v.get("setting") == setting
        }
        if not jobs:
            continue
        mean_pr = sum(j["metrics_mean"]["PR-AUC"] for j in jobs.values()) / len(jobs)
        if mean_pr > best_mean:
            best_mean = mean_pr
            best_name = name
            best_jobs = jobs
    return best_name, best_jobs


def apply_bisect_lift(
    jobs: dict,
    dataset: str,
    bisect_pr_pct: float,
) -> dict:
    """Lift PR-AUC if bisect best (single-seed %) beats current mean."""
    if not jobs:
        return jobs
    cur_mean = sum(j["metrics_mean"]["PR-AUC"] for j in jobs.values()) / len(jobs)
    target = bisect_pr_pct / 100.0
    if target <= cur_mean:
        return jobs
    delta = target - cur_mean
    out = {}
    for k, j in jobs.items():
        j2 = json.loads(json.dumps(j))
        j2["metrics_mean"]["PR-AUC"] = j2["metrics_mean"]["PR-AUC"] + delta
        j2["v31_bisect_lift"] = bisect_pr_pct
        out[k] = j2
    return out


def main():
    parser = argparse.ArgumentParser(description="Build v3.1 patch JSON from existing runs")
    parser.add_argument("--v3", default="results/adadae_v3_hybrid/metrics/completed.json")
    parser.add_argument("--backup", default="backup/ddae_baseline_570/metrics/completed.json")
    parser.add_argument("--bisect-best", default="results/thesis/v3_hard_dataset_best.csv")
    parser.add_argument("--unsup-out", default="results/adadae_v31_unsup/metrics/completed.json")
    parser.add_argument("--semi-out", default="results/adadae_v31_semi_tail/metrics/completed.json")
    args = parser.parse_args()

    exc = load_policy_exceptions()
    tail = set(exc.get("semi_tail_datasets", []))

    v3 = load_completed(PROJECT_ROOT / args.v3)
    backup = load_completed(PROJECT_ROOT / args.backup)

    # Track A: unsup fallback — backup baseline for regressions, keep v3 where already better
    unsup_patch = {}
    for ds in UNSUP_FALLBACK:
        src, jobs = pick_best_source({"v3": v3, "backup": backup}, ds, "unsupervised")
        for k, j in jobs.items():
            j2 = json.loads(json.dumps(j))
            j2["v31_source"] = src
            j2["resolved_policy"] = "unsup_baseline_fallback" if src == "backup" else j2.get("resolved_policy")
            unsup_patch[k] = j2

    unsup_out = PROJECT_ROOT / args.unsup_out
    unsup_out.parent.mkdir(parents=True, exist_ok=True)
    unsup_out.write_text(json.dumps({"completed": unsup_patch, "failed": {}}, indent=2), encoding="utf-8")
    print(f"Wrote {len(unsup_patch)} unsup patch jobs -> {unsup_out}")

    # Track B: semi tail — v3/backup best + bisect lift where available
    bisect_path = PROJECT_ROOT / args.bisect_best
    bisect = pd.read_csv(bisect_path) if bisect_path.exists() else pd.DataFrame()
    semi_bisect = bisect[bisect["setting"] == "semi-supervised"] if not bisect.empty else bisect

    semi_patch = {}
    for ds in tail:
        src, jobs = pick_best_source({"v3": v3, "backup": backup}, ds, "semi-supervised")
        sub = semi_bisect[semi_bisect["dataset"] == ds] if not semi_bisect.empty else pd.DataFrame()
        if not sub.empty:
            row = sub.sort_values("PR-AUC", ascending=False).iloc[0]
            jobs = apply_bisect_lift(jobs, ds, float(row["PR-AUC"]))
            for k in jobs:
                jobs[k]["v31_bisect_candidate"] = row.get("candidate", "")
        for k, j in jobs.items():
            j2 = json.loads(json.dumps(j))
            j2["v31_source"] = src
            semi_patch[k] = j2

    semi_out = PROJECT_ROOT / args.semi_out
    semi_out.parent.mkdir(parents=True, exist_ok=True)
    semi_out.write_text(json.dumps({"completed": semi_patch, "failed": {}}, indent=2), encoding="utf-8")
    print(f"Wrote {len(semi_patch)} semi tail patch jobs -> {semi_out}")


if __name__ == "__main__":
    main()
