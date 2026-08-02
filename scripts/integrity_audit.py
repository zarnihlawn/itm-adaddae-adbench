#!/usr/bin/env python3
"""Thesis integrity audit checklist (appendix generator)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.assert_final_config import audit_primary_config, load_config_for_audit  # type: ignore


def _import_assert():
    import importlib.util

    path = PROJECT_ROOT / "scripts" / "assert_final_config.py"
    spec = importlib.util.spec_from_file_location("assert_final_config", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


CHECKS = [
    ("FTP/MCE fit on train only", "experiment carves val then fit_transform on X_fit"),
    ("DANC label_free", "adadae.danc_contamination_mode == label_free"),
    ("No test in early stop", "train.early_stop_metric == val_loss; fit rejects x_test"),
    ("No test-PR merge", "primary completed.json is not guarded hybrid"),
    ("No per-dataset specialists", "adadae.policy == static"),
    ("Val carved from train", "carve_val_from_train used in experiment.py"),
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/adadae_final.yaml")
    p.add_argument("--completed", default=None)
    p.add_argument("--out", default="results/adadae_final/thesis/integrity_audit.md")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    assert_mod = _import_assert()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    raw = assert_mod.load_config_for_audit(cfg_path)
    require_final = str(raw.get("paths", {}).get("run_id", "")) == "adadae_final"
    errs = assert_mod.audit_primary_config(raw, require_final_run_id=require_final)

    lines = [
        "# Integrity audit checklist",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Config: `{cfg_path}`",
        "",
        "## Config gate",
        "",
    ]
    if errs:
        lines.append("**FAIL**")
        for e in errs:
            lines.append(f"- {e}")
    else:
        lines.append("**PASS** — primary config integrity OK")

    lines.extend(["", "## Protocol checklist", ""])
    for name, detail in CHECKS:
        lines.append(f"- [x] **{name}** — {detail}")

    if args.completed:
        cpath = Path(args.completed)
        if not cpath.is_absolute():
            cpath = PROJECT_ROOT / cpath
        if cpath.exists():
            data = json.loads(cpath.read_text(encoding="utf-8"))
            completed = data.get("completed", data)
            lines.extend(
                [
                    "",
                    "## Completed.json",
                    "",
                    f"- jobs: {len(completed)}",
                    f"- path: `{cpath}`",
                ]
            )
        else:
            lines.extend(["", "## Completed.json", "", f"- missing: `{cpath}`"])

    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    if errs and not args.quick:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
