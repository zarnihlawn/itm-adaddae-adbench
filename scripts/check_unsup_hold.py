#!/usr/bin/env python3
"""Loop 9: unsupervised hold check vs paper after PER config changes.

Fails if unsup macro PR or ROC drops below paper (or optional prior PER snapshot).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_gates import PUBLISHED, load_completed, macro_mean_metrics  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--completed",
        default="results/adadae_per/metrics/completed.json",
    )
    p.add_argument(
        "--min-unsup-pr",
        type=float,
        default=PUBLISHED["unsupervised"]["PR-AUC"],
        help="Must strictly beat this PR (default: paper)",
    )
    p.add_argument(
        "--min-unsup-roc",
        type=float,
        default=PUBLISHED["unsupervised"]["ROC-AUC"],
    )
    args = p.parse_args()
    path = Path(args.completed)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        print(f"MISSING {path}")
        return 1
    completed = load_completed(path)
    m = macro_mean_metrics(completed, "unsupervised")
    pr_ok = m["PR-AUC"] > args.min_unsup_pr
    roc_ok = m["ROC-AUC"] > args.min_unsup_roc
    report = {
        "pass": pr_ok and roc_ok,
        "unsupervised": m,
        "floor_PR": args.min_unsup_pr,
        "floor_ROC": args.min_unsup_roc,
        "delta_PR": m["PR-AUC"] - args.min_unsup_pr,
        "delta_ROC": m["ROC-AUC"] - args.min_unsup_roc,
        "note": "Reject any change that drops unsup below paper.",
    }
    out = path.parent.parent / "thesis" / "loop9_unsup_hold.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
