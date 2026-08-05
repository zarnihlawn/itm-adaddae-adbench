#!/usr/bin/env python3
"""Audit |AP − PR-AUC| gaps and quarantine v31-bisect inflated hybrids.

Usage:
  python scripts/audit_ap_pr_consistency.py --completed results/adadae_per/metrics/completed.json
  python scripts/audit_ap_pr_consistency.py --completed results/adadae_v51_hybrid/metrics/completed.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_gates import check_ap_pr_consistency, load_completed  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--completed", required=True)
    p.add_argument("--max-gap-pp", type=float, default=6.0)
    p.add_argument("--out", default=None, help="Optional JSON report path")
    p.add_argument(
        "--allow-bisect",
        action="store_true",
        help="Do not fail solely due to v31_bisect markers (still report them)",
    )
    args = p.parse_args()
    path = Path(args.completed)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    completed = load_completed(path)
    report = check_ap_pr_consistency(
        completed,
        max_gap_pp=args.max_gap_pp,
        quarantine_bisect=not args.allow_bisect,
    )
    report["completed_path"] = str(path)
    out = Path(args.out) if args.out else path.parent.parent / "thesis" / "ap_pr_consistency.json"
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"n_checked={report['n_checked']} n_bad={report['n_bad']} "
          f"n_bisect={report['n_bisect_markers']} pass={report['pass']}")
    for ex in report.get("examples", [])[:10]:
        print(f"  GAP {ex['key']}: AP={ex['AP']:.2f} PR={ex['PR-AUC']:.2f} "
              f"gap={ex['gap_pp']:.2f} bisect={ex.get('v31_bisect_candidate')}")
    print(f"Wrote {out}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
