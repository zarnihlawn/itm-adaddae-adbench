#!/usr/bin/env python3
"""Detect GPU VRAM and suggest AdaDDAE hardware profile."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    HARDWARE_PROFILES,
    detect_vram_gb,
    suggest_hardware_profile,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Suggest hardware profile from nvidia-smi VRAM")
    parser.add_argument(
        "--hardware",
        choices=sorted(HARDWARE_PROFILES.keys()),
        default=None,
        help="Override auto-detect with a named tier (8gb, 12gb, 16gb, rtx5070ti, ...)",
    )
    args = parser.parse_args()

    vram = detect_vram_gb()
    profile = args.hardware or suggest_hardware_profile()
    if args.hardware:
        from src.config import resolve_hardware_name

        profile = resolve_hardware_name(args.hardware)

    if vram is not None:
        print(f"detected_vram_gb={vram:.1f}")
    else:
        print("detected_vram_gb=unknown")
    print(f"suggested_profile={profile}")
    print(f"cli_flag=--hardware {profile.replace('.yaml', '').replace('hardware_', '')}")


if __name__ == "__main__":
    main()
