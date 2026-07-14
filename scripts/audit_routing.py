#!/usr/bin/env python3
"""Dry-run policy routing for all registry datasets; fail on category violations."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import build_registry
from src.policy import load_policy_exceptions, resolve_policy_name
from src.policy_meta import _is_valid_override


def main():
    parser = argparse.ArgumentParser(description="Audit routed policies for all datasets")
    parser.add_argument("--adbench", default="../ADBench/adbench/datasets")
    parser.add_argument("--fail-on-error", action="store_true", default=True)
    args = parser.parse_args()

    adbench = (PROJECT_ROOT / args.adbench).resolve()
    if not adbench.exists():
        print(f"ADBench not found: {adbench}")
        sys.exit(1)

    registry = build_registry(adbench)
    exc = load_policy_exceptions()
    violations: list[str] = []
    rows: list[str] = []

    for spec in sorted(registry, key=lambda s: (s.category, s.name)):
        meta = {"n": 1000.0, "d": 10.0}
        for setting in ["unsupervised", "semi-supervised"]:
            policy = resolve_policy_name(
                setting=setting,
                category=spec.category,
                dataset_name=spec.name,
                meta=meta,
                exceptions=exc,
            )
            ok = _is_valid_override(setting, spec.category, policy)
            status = "OK" if ok else "VIOLATION"
            rows.append(f"{status} {setting:18s} {spec.category:10s} {spec.name:20s} -> {policy}")
            if not ok:
                violations.append(f"{spec.name}/{setting}: {policy} invalid for {spec.category}")

    print("=== Routing audit ===")
    for r in rows:
        print(r)

    print(f"\nTotal: {len(rows)} routes, {len(violations)} violations")
    if violations:
        print("\nViolations:")
        for v in violations:
            print(f"  {v}")
        if args.fail_on_error:
            sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
