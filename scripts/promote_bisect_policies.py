#!/usr/bin/env python3
"""Promote bisect matrix winners into policy_exceptions.yaml (regression-guarded)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_v31_patches import (
    SEMI_NLP_FREEZE,
    best_bisect_candidate,
    load_completed,
    pick_best_source,
)

CANDIDATE_TO_POLICY_SEMI = {
    "baseline_ddae": "baseline_ddae",
    "semi_nlp_danc": "semi_nlp_baseline",
    "semi_speech_specialist": "semi_speech_specialist",
    "semi_rdt_tail": "semi_rdt_tail",
    "semi_cvnlp_ftp": "semi_cvnlp_ftp",
    "semi_cvnlp_taps_004": "semi_cvnlp_taps_light",
    "semi_cvnlp_taps_006": "semi_cvnlp_taps_light",
    "ssts": "baseline_ddae",
    "taps": "semi_cvnlp_taps_light",
    "lfdanc": "baseline_ddae",
    "ftp": "semi_cvnlp_ftp",
    "rdt": "semi_rdt_tail",
}

CANDIDATE_TO_POLICY_UNSUP = {
    "unsup_baseline": "unsup_baseline_fallback",
    "unsup_lfdanc": "unsup_ssts",
    "unsup_ssts": "unsup_ssts",
    "unsup_nlp_ssts_light": "unsup_nlp_ssts_light",
    "unsup_classical_plus": "unsup_classical_plus",
    "ddae_repro": "unsup_baseline_fallback",
    "taps": "unsup_ssts",
    "lfdanc": "unsup_ssts",
    "ssts": "unsup_ssts",
}


def main():
    parser = argparse.ArgumentParser(description="Promote bisect winners to policy_exceptions.yaml")
    parser.add_argument("--matrix", default="results/thesis/v31_semi_tail_matrix.csv")
    parser.add_argument("--unsup-matrix", default="results/thesis/v4_unsup_bisect_matrix.csv")
    parser.add_argument("--v3", default="results/adadae_v3_hybrid/metrics/completed.json")
    parser.add_argument("--backup", default="backup/ddae_baseline_570/metrics/completed.json")
    parser.add_argument("--out", default="configs/policy_exceptions.yaml")
    parser.add_argument("--guard-epsilon", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_path = PROJECT_ROOT / args.out
    with open(out_path, "r", encoding="utf-8") as f:
        exc = yaml.safe_load(f) or {}

    v3 = load_completed(PROJECT_ROOT / args.v3)
    backup = load_completed(PROJECT_ROOT / args.backup)
    semi_specialists = dict(exc.get("semi_specialists", {}))
    promotions: list[str] = []

    matrix_path = PROJECT_ROOT / args.matrix
    if matrix_path.exists():
        matrix = pd.read_csv(matrix_path)
        semi = matrix[matrix["setting"] == "semi-supervised"]
        tail = set(exc.get("semi_tail_datasets", []))

        for ds in sorted(tail):
            if ds in SEMI_NLP_FREEZE:
                continue
            cand, bisect_mean = best_bisect_candidate(semi, ds, "semi-supervised", "mean")
            if not cand:
                continue
            _, _, v3_mean = pick_best_source({"v3": v3}, ds, "semi-supervised")
            _, _, bk_mean = pick_best_source({"backup": backup}, ds, "semi-supervised")
            if bisect_mean <= max(v3_mean, bk_mean) + args.guard_epsilon:
                continue
            policy = CANDIDATE_TO_POLICY_SEMI.get(cand, cand)
            semi_specialists[ds] = policy
            promotions.append(f"semi {ds}: {cand} -> {policy} (mean PR {bisect_mean:.2f}%)")

    unsup_fallback = list(exc.get("unsup_baseline_fallback", []))
    unsup_matrix_path = PROJECT_ROOT / args.unsup_matrix
    if unsup_matrix_path.exists():
        matrix = pd.read_csv(unsup_matrix_path)
        unsup = matrix[matrix["setting"] == "unsupervised"]
        for ds in unsup["dataset"].unique():
            cand, bisect_mean = best_bisect_candidate(unsup, ds, "unsupervised", "mean")
            if not cand:
                continue
            _, _, v3_mean = pick_best_source({"v3": v3}, ds, "unsupervised")
            _, _, bk_mean = pick_best_source({"backup": backup}, ds, "unsupervised")
            if bisect_mean <= max(v3_mean, bk_mean) + args.guard_epsilon:
                continue
            policy = CANDIDATE_TO_POLICY_UNSUP.get(cand, cand)
            if policy == "unsup_baseline_fallback" and ds not in unsup_fallback:
                unsup_fallback.append(ds)
                promotions.append(f"unsup {ds}: fallback (mean PR {bisect_mean:.2f}%)")

    exc["semi_specialists"] = semi_specialists
    exc["unsup_baseline_fallback"] = sorted(set(unsup_fallback))

    print("=== Promotions ===")
    for p in promotions:
        print(f"  {p}")
    if not promotions:
        print("  (none — all guarded or no matrix)")

    if not args.dry_run:
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(exc, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        print(f"Wrote {out_path}")
    else:
        print("Dry run — no file written")


if __name__ == "__main__":
    main()
