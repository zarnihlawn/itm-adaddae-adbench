#!/usr/bin/env python3
"""Aggregate completed.json metrics by ADBench regime tags."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--completed", required=True)
    p.add_argument("--regimes", default="results/thesis/adbench_regimes.csv")
    p.add_argument("--out", default="results/thesis/regime_breakdown.json")
    args = p.parse_args()

    completed = json.loads(Path(args.completed).read_text())
    rows = list(csv.DictReader(Path(args.regimes).open(encoding="utf-8")))
    by_name = {r["name"]: r for r in rows}

    buckets: dict = defaultdict(list)
    for key, summary in completed.get("completed", {}).items():
        # key like cardio__semi-supervised__111
        name = key.split("__")[0]
        setting = key.split("__")[1] if "__" in key else ""
        meta = by_name.get(name, {})
        tags = (meta.get("regime_tags") or "mid").split(",")
        pr = (summary.get("metrics_mean") or {}).get("PR-AUC")
        if pr is None:
            continue
        for t in tags:
            buckets[f"{t}|{setting}"].append(float(pr))
        buckets[f"all|{setting}"].append(float(pr))
        buckets[f"cat:{meta.get('category', '?')}|{setting}"].append(float(pr))

    out = {}
    for k, vals in sorted(buckets.items()):
        out[k] = {
            "n": len(vals),
            "mean_pr": sum(vals) / len(vals),
            "min_pr": min(vals),
            "max_pr": max(vals),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({len(out)} buckets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
