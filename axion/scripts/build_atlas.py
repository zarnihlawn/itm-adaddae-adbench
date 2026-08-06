#!/usr/bin/env python3
"""CLI: build data/atlas_57.csv."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axion.data.atlas import build_atlas  # noqa: E402
from axion.paths import ATLAS_CSV, DEFAULT_ADBENCH_DATASETS  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Build AXION ADBench atlas_57.csv")
    p.add_argument(
        "--adbench-root",
        type=Path,
        default=DEFAULT_ADBENCH_DATASETS,
        help="Path to ADBench adbench/datasets",
    )
    p.add_argument("--out", type=Path, default=ATLAS_CSV)
    args = p.parse_args()
    df = build_atlas(adbench_root=args.adbench_root, out_csv=args.out)
    print(f"OK: {args.out} ({len(df)} datasets)")
    print(df.groupby("modality")["name"].count().to_string())


if __name__ == "__main__":
    main()
