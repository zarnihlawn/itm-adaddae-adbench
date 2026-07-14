#!/usr/bin/env python3
"""Update policy_exceptions.yaml from v3 hard-dataset bisect results."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BISECT = PROJECT_ROOT / "results/thesis/v3_hard_dataset_best.csv"
DEFAULT_EXCEPTIONS = PROJECT_ROOT / "configs/policy_exceptions.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bisect-csv", default=str(DEFAULT_BISECT))
    parser.add_argument("--out", default=str(DEFAULT_EXCEPTIONS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    bisect_path = Path(args.bisect_csv)
    if not bisect_path.is_absolute():
        bisect_path = PROJECT_ROOT / bisect_path
    if not bisect_path.exists():
        raise SystemExit(f"Bisect results not found: {bisect_path}")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path

    with open(out_path, "r", encoding="utf-8") as f:
        exc = yaml.safe_load(f) or {}

    best = pd.read_csv(bisect_path)
    changes = []

    for setting, fallback_key, candidates in [
        ("unsupervised", "unsup_baseline_fallback", {"baseline_ddae", "unsup_baseline", "ddae_repro"}),
        ("semi-supervised", "semi_nlp_baseline", {"baseline_ddae", "semi_nlp_baseline"}),
    ]:
        sub = best[best["setting"] == setting]
        for ds in sub["dataset"].unique():
            row = sub[sub["dataset"] == ds].iloc[0]
            if row["candidate"] in candidates:
                lst = exc.setdefault(fallback_key, [])
                if ds not in lst:
                    lst.append(ds)
                    changes.append(f"{setting}/{ds} -> {fallback_key} ({row['candidate']})")

        specialists = exc.setdefault("semi_specialists", {})
        semi_sub = best[best["setting"] == "semi-supervised"]
        for ds in semi_sub["dataset"].unique():
            row = semi_sub[semi_sub["dataset"] == ds].iloc[0]
            if str(row["candidate"]).startswith("semi_") and "specialist" in str(row["candidate"]):
                specialists[ds] = str(row["candidate"])
                changes.append(f"semi/{ds} -> specialist {row['candidate']}")

    print("Policy exception updates:")
    for c in changes:
        print(f"  {c}")
    if not changes:
        print("  (no changes)")

    if not args.dry_run:
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(exc, f, default_flow_style=False, sort_keys=False)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
