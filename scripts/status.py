#!/usr/bin/env python3
"""Print full-protocol progress from completed.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
path = PROJECT_ROOT / "results" / "metrics" / "completed.json"
if not path.exists():
    print("No completed.json yet")
    sys.exit(0)

state = json.loads(path.read_text())
done = state.get("completed", {})
failed = state.get("failed", {})
stale_failed = {k for k in failed if k in done}
active_failed = {k: v for k, v in failed.items() if k not in done}
print(f"Completed: {len(done)} / 570")
print(f"Failed (still pending retry): {len(active_failed)}")
if stale_failed:
    print(f"Stale failed entries (already completed): {len(stale_failed)}")
by_setting = {}
for k, v in done.items():
    by_setting.setdefault(v.get("setting", "?"), []).append(v)
for s, rows in by_setting.items():
    pr = sum(r["metrics_mean"]["PR-AUC"] for r in rows) / len(rows)
    roc = sum(r["metrics_mean"]["ROC-AUC"] for r in rows) / len(rows)
    print(f"  {s}: n={len(rows)} mean PR-AUC={pr*100:.2f} ROC-AUC={roc*100:.2f}")
if active_failed:
    print("Failures:", list(active_failed.keys())[:10])
    err0 = next(iter(active_failed.values()))
    print("Sample error:", err0[:120])
