#!/usr/bin/env python3
"""Write Vast run card for integrity + AdaDDAE-2 570 protocols."""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _load_assert():
    path = PROJECT_ROOT / "scripts" / "assert_final_config.py"
    spec = importlib.util.spec_from_file_location("assert_final_config", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    assert_mod = _load_assert()
    configs = [
        ("configs/adadae_final.yaml", False),
        ("configs/baselines_ddae_valstop.yaml", True),
        ("configs/adadae2_final.yaml", True),
    ]
    status = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "configs": {},
        "command": [
            "bash scripts/setup_vast.sh 16gb",
            "bash scripts/run_adadae_final_protocol.sh all 16gb",
            "bash scripts/run_adadae2_protocol.sh all 16gb",
            "bash scripts/sync_results_from_vast.sh <ssh-host>",
        ],
        "local_cuda": False,
        "note": "Full 570 jobs require Vast GPU; local machine prepares configs/smoke only.",
    }
    try:
        import torch

        status["local_cuda"] = bool(torch.cuda.is_available())
    except Exception:
        pass

    ok = True
    for rel, allow in configs:
        path = PROJECT_ROOT / rel
        if not path.exists():
            status["configs"][rel] = {"ok": False, "errors": ["missing file"]}
            ok = False
            continue
        raw = assert_mod.load_config_for_audit(path)
        errs = assert_mod.audit_primary_config(raw, require_final_run_id=not allow)
        status["configs"][rel] = {"ok": len(errs) == 0, "errors": errs}
        if errs:
            ok = False

    out = PROJECT_ROOT / "results" / "adadae_final" / "thesis" / "vast_run_card.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    print(f"Wrote {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
