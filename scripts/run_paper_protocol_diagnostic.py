#!/usr/bin/env python3
"""Loop 3: paper-protocol DDAE diagnostic (AnoDDAE-faithful, not a ship claim).

Runs configs/ddae_paper_protocol.yaml on a probe set (or --datasets) and
compares macro PR to published paper + fair valstop.

Usage:
  python scripts/run_paper_protocol_diagnostic.py --datasets Wilt glass cardio --seeds 111
  python scripts/run_paper_protocol_diagnostic.py --compare-only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PAPER_SEMI_PR = 61.36
PAPER_SEMI_ROC = 83.17
DEFAULT_PROBE = [
    "Wilt",
    "glass",
    "cardio",
    "WBC",
    "vertebral",
    "Hepatitis",
    "mammography",
]


def _macro(completed: dict, setting: str = "semi-supervised") -> dict:
    from collections import defaultdict
    import statistics as stats

    by = defaultdict(list)
    by_roc = defaultdict(list)
    for job in completed.values():
        if job.get("setting") != setting:
            continue
        m = job.get("metrics_mean") or job.get("metrics") or {}
        pr = m.get("PR-AUC")
        roc = m.get("ROC-AUC")
        if pr is None:
            continue
        pr_pp = float(pr) * 100 if float(pr) <= 1.5 else float(pr)
        by[job["dataset"]].append(pr_pp)
        if roc is not None:
            roc_pp = float(roc) * 100 if float(roc) <= 1.5 else float(roc)
            by_roc[job["dataset"]].append(roc_pp)
    if not by:
        return {"PR": float("nan"), "ROC": float("nan"), "n": 0}
    return {
        "PR": float(stats.mean(stats.mean(v) for v in by.values())),
        "ROC": float(stats.mean(stats.mean(v) for v in by_roc.values())) if by_roc else float("nan"),
        "n": len(by),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=[111])
    p.add_argument("--hardware", default=None)
    p.add_argument("--compare-only", action="store_true")
    p.add_argument(
        "--completed",
        default="results/ddae_paper_protocol/metrics/completed.json",
    )
    args = p.parse_args()
    completed_path = Path(args.completed)
    if not completed_path.is_absolute():
        completed_path = PROJECT_ROOT / completed_path

    if not args.compare_only:
        ds = args.datasets or DEFAULT_PROBE
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_full_protocol.py"),
            "--config",
            "configs/ddae_paper_protocol.yaml",
            "--datasets",
            *ds,
            "--seeds",
            *[str(s) for s in args.seeds],
        ]
        if args.hardware:
            cmd.extend(["--hardware", args.hardware])
        print("Running:", " ".join(cmd))
        rc = subprocess.call(cmd, cwd=str(PROJECT_ROOT))
        if rc != 0:
            return rc

    out = {
        "note": "Diagnostic only — paper protocol (100ep, no val carve, full-sum). Not ship claim.",
        "paper_semi": {"PR": PAPER_SEMI_PR, "ROC": PAPER_SEMI_ROC},
    }
    if completed_path.exists():
        data = json.loads(completed_path.read_text())
        completed = data.get("completed", data)
        out["paper_protocol"] = _macro(completed)
    fair = PROJECT_ROOT / "results/ddae_baseline_valstop/metrics/completed.json"
    if fair.exists():
        data = json.loads(fair.read_text())
        completed = data.get("completed", data)
        # Restrict to same datasets if probe results exist
        out["fair_valstop"] = _macro(completed)

    report = PROJECT_ROOT / "results/ddae_paper_protocol/thesis/loop3_diagnostic.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"Wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
