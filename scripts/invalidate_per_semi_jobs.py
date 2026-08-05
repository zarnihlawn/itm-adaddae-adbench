#!/usr/bin/env python3
"""Invalidate PER completed jobs whose routing/upgrades changed (Loop 2/4/7).

Removes matching keys from completed.json and per-job metric files so
`run_full_protocol.py` will retrain them. Does not delete unrelated jobs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Datasets whose semi (and sometimes unsup) recipes changed in Loop 2/7
INVALIDATE_SEMI = {
    "Wilt",
    "glass",
    "vertebral",
    "CIFAR10",
    "Waveform",
    "speech",
    "cover",
    "optdigits",
    "wine",
    "Hepatitis",
    "fraud",
    "smtp",
    "SVHN",
    "SpamBase",
    "satellite",
    "yeast",
}
# Horizon / val_fraction_semi affects ALL semi jobs — invalidate all semi
INVALIDATE_ALL_SEMI = True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--completed",
        default="results/adadae_per/metrics/completed.json",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    path = Path(args.completed)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        print(f"MISSING {path}")
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    completed = data.get("completed", data)
    metrics_dir = path.parent
    remove_keys = []
    for key in list(completed.keys()):
        parts = key.split("__")
        if len(parts) < 3:
            continue
        ds, setting, _seed = parts[0], parts[1], parts[2]
        if setting == "semi-supervised" and (
            INVALIDATE_ALL_SEMI or ds in INVALIDATE_SEMI
        ):
            remove_keys.append(key)
        elif setting == "unsupervised" and ds in {"speech", "ALOI", "optdigits"}:
            # GATE unchanged; keep unless we want full refresh — skip
            pass

    print(f"Will invalidate {len(remove_keys)} / {len(completed)} jobs")
    if args.dry_run:
        for k in remove_keys[:20]:
            print(f"  {k}")
        return 0

    for key in remove_keys:
        completed.pop(key, None)
        mf = metrics_dir / f"{key}.json"
        if mf.exists():
            mf.unlink()
    data["completed"] = completed
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Remaining {len(completed)} jobs in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
