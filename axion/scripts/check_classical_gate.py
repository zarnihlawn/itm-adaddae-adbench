#!/usr/bin/env python3
"""Hard classical gate before expensive full probes.

Exit 0 only if latest MACRO in probe_loop.jsonl (or probe_summary.json) has:
  semi-supervised PR-AUC >= --min-semi-pr  (default 60)
  and optionally both pass_probe_margin flags.

Examples:
  python scripts/check_classical_gate.py --run-id axion_g4_classical
  python scripts/check_classical_gate.py --run-id axion_g4_classical --require-margin
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axion.config import load_config  # noqa: E402


def _load_macro(run_dir: Path) -> dict:
    loop = run_dir / "probe_loop.jsonl"
    if loop.exists():
        rows = [json.loads(l) for l in loop.read_text().splitlines() if l.strip()]
        if rows:
            return rows[-1].get("probe_macro") or {}
    summary = run_dir / "probe_summary.json"
    if summary.exists():
        return (json.loads(summary.read_text()).get("probe_macro")) or {}
    raise SystemExit(f"No probe_loop.jsonl or probe_summary.json under {run_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="AXION classical probe gate")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--run-id", default="axion_g4_classical")
    p.add_argument("--min-semi-pr", type=float, default=60.0)
    p.add_argument(
        "--require-margin",
        action="store_true",
        help="Also require pass_probe_margin on both settings",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    run_dir = Path(cfg["paths"]["results_dir"]) / args.run_id
    if not run_dir.exists():
        raise SystemExit(f"MISSING gate artifacts: {run_dir} (run classical probe first)")

    macro = _load_macro(run_dir)
    if "semi-supervised" not in macro or "unsupervised" not in macro:
        raise SystemExit(f"Incomplete probe_macro in {run_dir}: keys={list(macro)}")

    semi = macro["semi-supervised"]
    unsup = macro["unsupervised"]
    semi_pr = float(semi["PR-AUC"])
    unsup_pr = float(unsup["PR-AUC"])
    print(
        f"GATE {args.run_id}: unsup PR={unsup_pr:.2f}  semi PR={semi_pr:.2f}  "
        f"(need semi ≥ {args.min_semi_pr:.1f})"
    )
    print(
        f"  unsup margin={unsup.get('pass_probe_margin')}  "
        f"semi margin={semi.get('pass_probe_margin')}"
    )

    ok = semi_pr >= float(args.min_semi_pr)
    if args.require_margin:
        ok = ok and bool(unsup.get("pass_probe_margin")) and bool(semi.get("pass_probe_margin"))

    if not ok:
        print("GATE FAIL — refusing full probe. Fix method / re-run classical only.")
        raise SystemExit(2)
    print("GATE PASS — full probe allowed.")


if __name__ == "__main__":
    main()
