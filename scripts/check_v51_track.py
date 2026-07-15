#!/usr/bin/env python3
"""Verify v5.1 track completed.json job counts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED = {
    "phase1": 570,
    "mce": 85,
    "smc": 20,
    "gate": 15,
}


def count_jobs(path: Path) -> int:
    if not path.exists():
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    return len(data.get("completed", data))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("track", choices=["phase1", "mce", "smc", "gate", "all"])
    parser.add_argument("--min-frac", type=float, default=0.5)
    args = parser.parse_args()

    paths = {
        "phase1": PROJECT_ROOT / "results/adadae_v51_phase1/metrics/completed.json",
        "mce": PROJECT_ROOT / "results/adadae_v51_mce/metrics/completed.json",
        "smc": PROJECT_ROOT / "results/adadae_v51_smc/metrics/completed.json",
        "gate": PROJECT_ROOT / "results/adadae_v51_gate/metrics/completed.json",
    }
    tracks = list(EXPECTED.keys()) if args.track == "all" else [args.track]
    ok = True
    for t in tracks:
        n = count_jobs(paths[t])
        exp = EXPECTED[t]
        min_n = int(exp * args.min_frac)
        status = "OK" if n >= min_n else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"{t}: {n} jobs (expected ~{exp}, min {min_n}) {status}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
