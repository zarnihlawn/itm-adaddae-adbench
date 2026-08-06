#!/usr/bin/env python3
"""Phase0 revoke audit: harmful A6 off + wine/census locked to baseline_ddae."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_yaml  # noqa: E402
from src.policy_per import apply_per_config, clear_per_upgrades_cache  # noqa: E402

# Must stay baseline_ddae — select/freeze must not re-apply RDT/champion
PHASE0_BASELINE_LOCKS = ("wine", "census")


def main() -> int:
    clear_per_upgrades_cache()
    cfg = load_yaml(PROJECT_ROOT / "configs" / "adadae_per.yaml")
    checks = []
    ok = True
    for ds, expect_protect in [
        ("smtp", True),
        ("donors", True),
        ("wine", True),
        ("census", True),
        ("speech", False),
        ("glass", False),
        ("Wilt", False),
        ("Agnews", False),
    ]:
        cat = "nlp" if ds == "Agnews" else "classical"
        out = apply_per_config(
            cfg, "semi-supervised", cat, ds, meta={"n": 1000.0, "d": 20.0}
        )
        ad = out["adadae"]
        pol = str(ad.get("resolved_policy", ""))
        row = {
            "dataset": ds,
            "resolved_policy": pol,
            "use_apex": bool(ad.get("use_apex")),
            "use_nautilus": bool(ad.get("use_nautilus")),
            "use_mce": bool(ad.get("use_mce")),
            "use_rejection_training": bool(ad.get("use_rejection_training")),
            "use_ftp": bool(ad.get("use_ftp")),
            "contrastive": bool((out.get("train") or {}).get("contrastive")),
            "protected": "protect" in pol,
        }
        if expect_protect:
            if ad.get("use_apex") or (ds == "wine" and ad.get("use_nautilus")):
                ok = False
                row["fail"] = "still has revoked A6"
            if "protect" not in pol and ds in ("smtp", "donors", "wine", "census"):
                ok = False
                row["fail"] = "missing protect tag"
        if ds in PHASE0_BASELINE_LOCKS:
            if "baseline_ddae" not in pol:
                ok = False
                row["fail"] = f"PHASE0_LOCK: expected baseline_ddae, got {pol}"
            if ad.get("use_rejection_training"):
                ok = False
                row["fail"] = "PHASE0_LOCK: RDT still on"
            if ad.get("use_ftp") and ds == "wine":
                # baseline_ddae must not enable FTP (champion_semi did)
                ok = False
                row["fail"] = "PHASE0_LOCK: FTP still on (champion leak)"
        if ds == "Agnews" and (out.get("train") or {}).get("contrastive"):
            ok = False
            row["fail"] = "Agnews TAPS still on (strip from contrastive_taps_semi)"
        checks.append(row)

    report = {
        "pass": ok,
        "revoked": [
            "smtp:apex",
            "wine:nautilus+champion/RDT→baseline_ddae",
            "census:RDT→baseline_ddae",
            "Agnews:taps stripped",
            "speech:mce+smc+apex kitchen-sink",
        ],
        "phase0_locks": {ds: "baseline_ddae" for ds in PHASE0_BASELINE_LOCKS},
        "checks": checks,
    }
    out_path = PROJECT_ROOT / "results/adadae_per/thesis/phase0_revoke.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
