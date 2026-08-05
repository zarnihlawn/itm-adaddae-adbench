#!/usr/bin/env python3
"""Invalidate PER completed jobs so run_full_protocol will retrain them.

Usage:
  # All semi jobs (285)
  python scripts/invalidate_per_semi_jobs.py --all-semi

  # Hard-tail semi keys only (~60)
  python scripts/invalidate_per_semi_jobs.py \\
    --datasets speech ALOI celeba SVHN CIFAR10 Wilt \\
              Imdb Amazon Yelp Agnews 20newsgroups census \\
    --settings semi-supervised

  python scripts/invalidate_per_semi_jobs.py --datasets speech Wilt --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HARD_TAIL_DEFAULT = [
    "speech",
    "ALOI",
    "celeba",
    "SVHN",
    "CIFAR10",
    "Wilt",
    "Imdb",
    "Amazon",
    "Yelp",
    "Agnews",
    "20newsgroups",
    "census",
]

BLEED_CLASSICAL = ["smtp", "satimage-2", "Pima", "Stamps", "letter", "wine"]

SHIP_PROBE = list(dict.fromkeys(HARD_TAIL_DEFAULT + BLEED_CLASSICAL))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--completed",
        default="results/adadae_per/metrics/completed.json",
    )
    p.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Only invalidate these dataset names (case-sensitive ADBench names)",
    )
    p.add_argument(
        "--settings",
        nargs="*",
        default=["semi-supervised"],
        help="Settings to invalidate (default: semi-supervised)",
    )
    p.add_argument(
        "--all-semi",
        action="store_true",
        help="Invalidate every semi-supervised job (ignores --datasets)",
    )
    p.add_argument(
        "--hard-tails",
        action="store_true",
        help="Shorthand: invalidate HARD_TAIL_DEFAULT semi jobs",
    )
    p.add_argument(
        "--bleed-classical",
        action="store_true",
        help="Invalidate BLEED_CLASSICAL semi jobs only",
    )
    p.add_argument(
        "--ship-probe",
        action="store_true",
        help="Invalidate hard-12 + bleed-classical semi (~90 jobs)",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    path = Path(args.completed)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        print(f"MISSING {path}")
        return 1

    settings = set(args.settings or ["semi-supervised"])
    if args.all_semi:
        datasets = None  # all datasets for selected settings
        settings = {"semi-supervised"}
    elif args.ship_probe:
        datasets = set(SHIP_PROBE)
    elif args.hard_tails and args.bleed_classical:
        datasets = set(HARD_TAIL_DEFAULT) | set(BLEED_CLASSICAL)
    elif args.hard_tails:
        datasets = set(HARD_TAIL_DEFAULT)
    elif args.bleed_classical:
        datasets = set(BLEED_CLASSICAL)
    elif args.datasets:
        datasets = set(args.datasets)
    else:
        print(
            "ERROR: pass --all-semi, --hard-tails, --bleed-classical, "
            "--ship-probe, or --datasets NAME ...",
            file=sys.stderr,
        )
        return 2

    data = json.loads(path.read_text(encoding="utf-8"))
    completed = data.get("completed", data)
    metrics_dir = path.parent
    remove_keys = []
    for key in list(completed.keys()):
        parts = key.split("__")
        if len(parts) < 3:
            continue
        ds, setting, _seed = parts[0], parts[1], parts[2]
        if setting not in settings:
            continue
        if datasets is not None and ds not in datasets:
            continue
        remove_keys.append(key)

    print(
        f"Will invalidate {len(remove_keys)} / {len(completed)} jobs "
        f"(settings={sorted(settings)}, "
        f"datasets={'ALL' if datasets is None else sorted(datasets)})"
    )
    if args.dry_run:
        for k in remove_keys[:30]:
            print(f"  {k}")
        if len(remove_keys) > 30:
            print(f"  ... +{len(remove_keys) - 30} more")
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
