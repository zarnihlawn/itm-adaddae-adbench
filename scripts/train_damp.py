#!/usr/bin/env python3
"""Train DAMP meta-policy from bisect matrices (LODO)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.policy_damp import train_and_export
from src.policy_meta import HOLDOUT_DEFAULT


def main():
    parser = argparse.ArgumentParser(description="Train DAMP LODO meta-policy")
    parser.add_argument(
        "--unsup-matrix",
        default="results/thesis/v4_unsup_bisect_matrix.csv",
    )
    parser.add_argument(
        "--semi-matrix",
        default="results/thesis/v31_semi_tail_matrix.csv",
    )
    parser.add_argument(
        "--fallback-matrix",
        default="results/thesis/v41_unsup_fallback_bisect_matrix.csv",
    )
    parser.add_argument("--out", default="configs/damp_model.pkl")
    args = parser.parse_args()

    paths = []
    for p in [args.unsup_matrix, args.semi_matrix, args.fallback_matrix]:
        fp = Path(p)
        if not fp.is_absolute():
            fp = PROJECT_ROOT / fp
        if fp.exists():
            paths.append(fp)

    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out

    report = train_and_export(matrix_paths=paths, holdout=HOLDOUT_DEFAULT, out_path=out)
    print("DAMP training report:")
    for k, v in report.items():
        if k != "holdout":
            print(f"  {k}: {v}")
    if "holdout" in report:
        h = report["holdout"]
        print(f"  holdout wins: {h.get('wins', 0)}/{h.get('n_holdout', 0)}")
    if report.get("error"):
        sys.exit(1)
    print(f"Model saved to {out}")


if __name__ == "__main__":
    main()
