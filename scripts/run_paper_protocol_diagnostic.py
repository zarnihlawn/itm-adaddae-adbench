#!/usr/bin/env python3
"""Loop 3: paper-protocol DDAE diagnostic (AnoDDAE-faithful, not a ship claim).

Runs configs/ddae_paper_protocol.yaml on a probe set (or --datasets) and
compares macro PR to published paper + fair valstop.

Also writes results/adadae_per/thesis/loop3_diagnostic.json and phase4 status.

Usage:
  python scripts/run_paper_protocol_diagnostic.py --wire-only
  python scripts/run_paper_protocol_diagnostic.py --datasets ... --seeds 111 222 --hardware 16gb
  python scripts/run_paper_protocol_diagnostic.py --compare-only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PAPER_SEMI_PR = 61.36
PAPER_SEMI_ROC = 83.17

# Hard + classical sample (protocol-tax probe)
DEFAULT_PROBE = [
    "speech",
    "ALOI",
    "celeba",
    "SVHN",
    "CIFAR10",
    "Wilt",
    "Imdb",
    "Agnews",
    "cardio",
    "glass",
    "WBC",
    "vertebral",
    "smtp",
    "Pima",
]


def _macro(completed: dict, setting: str = "semi-supervised", datasets=None) -> dict:
    from collections import defaultdict
    import statistics as stats

    allow = set(datasets) if datasets else None
    by = defaultdict(list)
    by_roc = defaultdict(list)
    for key, job in completed.items():
        if isinstance(job, dict) and job.get("setting"):
            setting_j = job.get("setting")
            ds = job.get("dataset")
        else:
            parts = str(key).split("__")
            if len(parts) < 3:
                continue
            ds, setting_j = parts[0], parts[1]
            job = job if isinstance(job, dict) else {}
        if setting_j != setting:
            continue
        if allow is not None and ds not in allow:
            continue
        m = job.get("metrics_mean") or job.get("metrics") or {}
        pr = m.get("PR-AUC")
        roc = m.get("ROC-AUC")
        if pr is None:
            continue
        pr_pp = float(pr) * 100 if float(pr) <= 1.5 else float(pr)
        by[ds].append(pr_pp)
        if roc is not None:
            roc_pp = float(roc) * 100 if float(roc) <= 1.5 else float(roc)
            by_roc[ds].append(roc_pp)
    if not by:
        return {"PR": float("nan"), "ROC": float("nan"), "n": 0, "datasets": []}
    return {
        "PR": float(stats.mean(stats.mean(v) for v in by.values())),
        "ROC": float(stats.mean(stats.mean(v) for v in by_roc.values())) if by_roc else float("nan"),
        "n": len(by),
        "datasets": sorted(by.keys()),
    }


def _write_status(out: dict, probe: list) -> None:
    paper_dir = PROJECT_ROOT / "results/ddae_paper_protocol/thesis"
    per_dir = PROJECT_ROOT / "results/adadae_per/thesis"
    paper_dir.mkdir(parents=True, exist_ok=True)
    per_dir.mkdir(parents=True, exist_ok=True)

    paper_path = paper_dir / "loop3_diagnostic.json"
    paper_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Mirror into PER thesis for ship docs
    per_loop3 = {
        "loop": 3,
        "status": out.get("status"),
        "timestamp": out.get("timestamp"),
        "note": out.get("note"),
        "command": out.get("command"),
        "hypothesis": out.get("hypothesis"),
        "paper_semi": out.get("paper_semi"),
        "fair_valstop": out.get("fair_valstop"),
        "paper_protocol": out.get("paper_protocol"),
        "probe_datasets": probe,
        "feeds": "Appendix only — not a ship claim. Ship stays integrity valstop + PER.",
    }
    (per_dir / "loop3_diagnostic.json").write_text(
        json.dumps(per_loop3, indent=2), encoding="utf-8"
    )

    phase4 = {
        "phase": 4,
        "status": out.get("status"),
        "config": "configs/ddae_paper_protocol.yaml",
        "script": "scripts/run_paper_protocol_diagnostic.py",
        "command": out.get("command"),
        "hypothesis": out.get("hypothesis"),
        "paper_semi": out.get("paper_semi"),
        "fair_valstop_semi": {
            "PR": (out.get("fair_valstop") or {}).get("PR"),
            "ROC": (out.get("fair_valstop") or {}).get("ROC"),
            "n": (out.get("fair_valstop") or {}).get("n"),
        },
        "paper_protocol": out.get("paper_protocol"),
        "interpretation": out.get("interpretation"),
    }
    (per_dir / "phase4_paper_diag.json").write_text(
        json.dumps(phase4, indent=2), encoding="utf-8"
    )
    print(f"Wrote {paper_path}")
    print(f"Wrote {per_dir / 'loop3_diagnostic.json'}")
    print(f"Wrote {per_dir / 'phase4_paper_diag.json'}")


def _interpretation(out: dict) -> str:
    pp = out.get("paper_protocol") or {}
    fair = out.get("fair_valstop") or {}
    if not pp or pp.get("n", 0) == 0:
        return (
            "pending_gpu: run diagnostic on Vast. "
            "If paper_protocol≈61.36, remaining ship gap is integrity tax; "
            "if ≈fair (~58.7), focus 100% on recipe wins."
        )
    pr = pp.get("PR")
    if pr != pr:  # NaN
        return "paper_protocol macro NaN — check completed jobs"
    if pr >= 60.5:
        return (
            f"paper_protocol PR={pr:.2f} near published 61.36 — "
            "expect ~1.5–2.5 irreducible under integrity; need big embed lifts."
        )
    if fair.get("PR") and abs(pr - float(fair["PR"])) < 1.0:
        return (
            f"paper_protocol PR={pr:.2f} ≈ fair — published table may differ "
            "for other reasons; focus on recipe wins under integrity."
        )
    return f"paper_protocol PR={pr:.2f} between fair and paper — partial protocol tax."


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=[111, 222])
    p.add_argument("--hardware", default=None)
    p.add_argument("--compare-only", action="store_true")
    p.add_argument(
        "--wire-only",
        action="store_true",
        help="Write pending_gpu status + command sheet without training",
    )
    p.add_argument(
        "--completed",
        default="results/ddae_paper_protocol/metrics/completed.json",
    )
    args = p.parse_args()
    completed_path = Path(args.completed)
    if not completed_path.is_absolute():
        completed_path = PROJECT_ROOT / completed_path

    probe = args.datasets or list(DEFAULT_PROBE)
    cmd_parts = [
        "python",
        "scripts/run_paper_protocol_diagnostic.py",
        "--datasets",
        *probe,
        "--seeds",
        *[str(s) for s in args.seeds],
    ]
    if args.hardware:
        cmd_parts.extend(["--hardware", args.hardware])
    else:
        cmd_parts.extend(["--hardware", "16gb"])
    command = " ".join(cmd_parts)

    hypothesis = (
        "Paper semi 61.36 may be closer to paper-protocol DDAE (100ep, no val carve, "
        "full-sum) than to fair valstop (~58.7); quantify tax before attributing all "
        "gap to PER recipes."
    )

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "note": "Diagnostic only — paper protocol (100ep, no val carve, full-sum). Not ship claim.",
        "hypothesis": hypothesis,
        "command": command,
        "probe_datasets": probe,
        "paper_semi": {"PR": PAPER_SEMI_PR, "ROC": PAPER_SEMI_ROC},
    }

    if args.wire_only:
        out["status"] = "wired_pending_gpu"
        fair = PROJECT_ROOT / "results/ddae_baseline_valstop/metrics/completed.json"
        if fair.exists():
            data = json.loads(fair.read_text())
            completed = data.get("completed", data)
            out["fair_valstop"] = _macro(completed, datasets=probe)
            # Also full fair macro for reference
            out["fair_valstop_full57"] = _macro(completed)
        out["paper_protocol"] = {"PR": None, "ROC": None, "n": 0, "datasets": []}
        out["interpretation"] = _interpretation(out)
        _write_status(out, probe)
        print(json.dumps(out, indent=2))
        return 0

    if not args.compare_only:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_full_protocol.py"),
            "--config",
            "configs/ddae_paper_protocol.yaml",
            "--datasets",
            *probe,
            "--seeds",
            *[str(s) for s in args.seeds],
        ]
        if args.hardware:
            cmd.extend(["--hardware", args.hardware])
        print("Running:", " ".join(cmd))
        rc = subprocess.call(cmd, cwd=str(PROJECT_ROOT))
        if rc != 0:
            return rc

    fair = PROJECT_ROOT / "results/ddae_baseline_valstop/metrics/completed.json"
    if fair.exists():
        data = json.loads(fair.read_text())
        completed = data.get("completed", data)
        out["fair_valstop"] = _macro(completed, datasets=probe)
        out["fair_valstop_full57"] = _macro(completed)

    if completed_path.exists():
        data = json.loads(completed_path.read_text())
        completed = data.get("completed", data)
        out["paper_protocol"] = _macro(completed, datasets=probe)
        out["status"] = "complete" if (out["paper_protocol"].get("n") or 0) > 0 else "empty_completed"
    else:
        out["paper_protocol"] = {"PR": None, "ROC": None, "n": 0, "datasets": []}
        out["status"] = "pending_gpu_no_completed"

    out["interpretation"] = _interpretation(out)
    _write_status(out, probe)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
