#!/usr/bin/env python3
"""Compute benchmark tier projections (T0–T4) vs +30% relative targets."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_gates import load_completed, macro_mean_metrics

PAPER = {"unsupervised": 32.77, "semi-supervised": 61.36}
HARD_TAIL_THRESHOLD = 50.0
REL_TARGET = 1.30


def per_dataset_pr(completed: dict, setting: str) -> pd.Series:
    rows = []
    for job in completed.values():
        if job.get("setting") != setting:
            continue
        rows.append({"dataset": job["dataset"], "PR": job["metrics_mean"]["PR-AUC"] * 100})
    if not rows:
        return pd.Series(dtype=float)
    return pd.DataFrame(rows).groupby("dataset")["PR"].mean()


def hard_tail_macro(pr_by_ds: pd.Series, threshold: float = HARD_TAIL_THRESHOLD) -> dict:
    hard = pr_by_ds[pr_by_ds < threshold]
    easy = pr_by_ds[pr_by_ds >= threshold]
    return {
        "n_hard": int(len(hard)),
        "n_easy": int(len(easy)),
        "hard_macro_pr": float(hard.mean()) if len(hard) else 0.0,
        "easy_macro_pr": float(easy.mean()) if len(easy) else 0.0,
        "hard_datasets": sorted(hard.index.tolist()),
    }


def scenario_hard_only_30pct(pr_by_ds: pd.Series) -> float:
    """If PR<50% datasets gain +30% rel (capped at 100%), easy unchanged."""
    out = []
    for pr in pr_by_ds:
        if pr < HARD_TAIL_THRESHOLD:
            out.append(min(100.0, pr * REL_TARGET))
        else:
            out.append(pr)
    return float(np.mean(out))


def bootstrap_ci(pr_by_ds: pd.Series, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    ds = pr_by_ds.values
    if len(ds) < 2:
        m = float(np.mean(ds)) if len(ds) else 0.0
        return m, m
    boots = [float(rng.choice(ds, size=len(ds), replace=True).mean()) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    parser = argparse.ArgumentParser(description="Compute benchmark tiers T0–T4")
    parser.add_argument("--completed", default="results/adadae_v41_hybrid/metrics/completed.json")
    parser.add_argument("--out", default="results/adadae_v5/thesis/eval_contract.json")
    parser.add_argument("--rel-target", type=float, default=REL_TARGET)
    args = parser.parse_args()

    completed_path = Path(args.completed)
    if not completed_path.is_absolute():
        completed_path = PROJECT_ROOT / completed_path
    completed = load_completed(completed_path)

    tiers: dict = {"baseline_path": str(completed_path), "rel_target": args.rel_target}
    for setting in ["unsupervised", "semi-supervised"]:
        macro = macro_mean_metrics(completed, setting)
        pr_ds = per_dataset_pr(completed, setting)
        ht = hard_tail_macro(pr_ds)
        cur = macro["PR-AUC"]
        target_full = cur * args.rel_target
        target_paper = PAPER[setting] * args.rel_target
        scenario_ht = scenario_hard_only_30pct(pr_ds)
        ci_lo, ci_hi = bootstrap_ci(pr_ds)

        tiers[setting] = {
            "T0_current_macro_pr": cur,
            "T0_macro_pr_ci_95": [ci_lo, ci_hi],
            "T1_routing_leftovers_est": cur + 0.2 if setting == "unsupervised" else cur + 0.15,
            "T2_hard_tail_30pct_rel_scenario": scenario_ht,
            "T3_v5_realistic_range": [cur + 5, cur + 9] if setting == "unsupervised" else [cur + 2, cur + 5],
            "T4_full_macro_30pct_rel_target": target_full,
            "T4_infeasible": target_full > 55 if setting == "unsupervised" else target_full > 70,
            "gap_to_T4_full_macro_pp": target_full - cur,
            "paper_pr": PAPER[setting],
            "paper_30pct_rel_target": target_paper,
            **ht,
        }

    unsup = tiers["unsupervised"]["T0_current_macro_pr"]
    semi = tiers["semi-supervised"]["T0_current_macro_pr"]
    tiers["combined"] = {
        "T0_current_macro_pr": (unsup + semi) / 2.0,
        "T2_hard_tail_30pct_rel_scenario": (
            tiers["unsupervised"]["T2_hard_tail_30pct_rel_scenario"]
            + tiers["semi-supervised"]["T2_hard_tail_30pct_rel_scenario"]
        )
        / 2.0,
        "T4_full_macro_30pct_rel_target": (
            tiers["unsupervised"]["T4_full_macro_30pct_rel_target"]
            + tiers["semi-supervised"]["T4_full_macro_30pct_rel_target"]
        )
        / 2.0,
    }

    tiers["evaluation_contract"] = {
        "primary_metrics": ["full_macro_pr", "hard_tail_macro_pr", "roc_auc"],
        "hard_tail_threshold_pct": HARD_TAIL_THRESHOLD,
        "report_both_full_and_hard_tail": True,
        "lodo_holdout": sorted(["speech", "Agnews", "Wilt", "celeba", "cardio"]),
    }

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(tiers, indent=2), encoding="utf-8")

    print("=== Benchmark tiers ===")
    for setting in ["unsupervised", "semi-supervised"]:
        t = tiers[setting]
        print(
            f"{setting}: T0={t['T0_current_macro_pr']:.2f}% "
            f"T2(hard+30%rel)={t['T2_hard_tail_30pct_rel_scenario']:.2f}% "
            f"T4(target)={t['T4_full_macro_30pct_rel_target']:.2f}% "
            f"infeasible={t['T4_infeasible']}"
        )
    print(f"Combined T0={tiers['combined']['T0_current_macro_pr']:.2f}%")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
