#!/usr/bin/env python3
"""Export routing_rules.yaml from bisect matrices."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.policy_meta import export_routing_rules

HOLDOUT = {"speech", "Agnews", "Wilt", "celeba", "cardio"}


def main():
    parser = argparse.ArgumentParser(description="Build routing_rules.yaml from bisect CSVs")
    parser.add_argument("--semi-matrix", default="results/thesis/v31_semi_tail_matrix.csv")
    parser.add_argument("--unsup-matrix", default="results/thesis/v4_unsup_bisect_matrix.csv")
    parser.add_argument("--out", default="configs/routing_rules.yaml")
    parser.add_argument("--max-depth", type=int, default=4)
    args = parser.parse_args()

    paths = []
    for p in [args.semi_matrix, args.unsup_matrix]:
        fp = PROJECT_ROOT / p
        if fp.exists():
            paths.append(fp)

    if not paths:
        print("No bisect matrices found — routing_rules.yaml stays disabled")
        sys.exit(0)

    rules = export_routing_rules(
        paths,
        PROJECT_ROOT / args.out,
        max_depth=args.max_depth,
        holdout=HOLDOUT,
    )
    print(f"Exported routing rules: enabled={rules.get('enabled')}")
    n_unsup = len(rules.get("dataset_overrides", {}).get("unsupervised", {}))
    n_semi = len(rules.get("dataset_overrides", {}).get("semi-supervised", {}))
    print(f"  unsup overrides: {n_unsup}, semi overrides: {n_semi}")


if __name__ == "__main__":
    main()
