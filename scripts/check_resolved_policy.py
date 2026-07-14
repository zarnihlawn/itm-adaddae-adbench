#!/usr/bin/env python3
"""Verify resolved_policy is stored in completed.json patch jobs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.policy import load_policy_exceptions, resolve_policy_name


def main():
    parser = argparse.ArgumentParser(description="Check resolved_policy in completed.json")
    parser.add_argument("--completed", required=True)
    parser.add_argument("--setting", default="unsupervised")
    parser.add_argument("--expect-fallback", nargs="*", default=None)
    args = parser.parse_args()

    path = Path(args.completed)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    data = json.loads(path.read_text(encoding="utf-8"))
    completed = data.get("completed", data)

    exc = load_policy_exceptions()
    fallback = set(exc.get("unsup_baseline_fallback", []))

    missing = []
    wrong = []
    for key, job in completed.items():
        if job.get("setting") != args.setting:
            continue
        ds = job.get("dataset", "")
        pol = job.get("resolved_policy")
        if pol is None:
            missing.append(ds)
            continue
        if args.expect_fallback and ds in args.expect_fallback:
            if pol != "unsup_baseline_fallback":
                wrong.append((ds, pol))

    print(f"Jobs ({args.setting}): {sum(1 for j in completed.values() if j.get('setting')==args.setting)}")
    print(f"Missing resolved_policy: {len(missing)}")
    if missing:
        print("  ", sorted(set(missing))[:20])
    print(f"Wrong fallback policy: {len(wrong)}")
    for ds, pol in wrong[:15]:
        print(f"  {ds}: got {pol}, expected unsup_baseline_fallback")
        expected = resolve_policy_name(args.setting, "classical", ds)
        print(f"    router says: {expected}")

    if args.setting == "unsupervised":
        for ds in sorted(fallback):
            jobs = [j for j in completed.values() if j.get("dataset") == ds and j.get("setting") == "unsupervised"]
            if not jobs:
                continue
            pols = {j.get("resolved_policy") for j in jobs}
            print(f"  {ds}: policies in file = {pols}")

    ok = len(missing) == 0 and len(wrong) == 0
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
