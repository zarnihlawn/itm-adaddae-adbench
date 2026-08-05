#!/usr/bin/env python3
"""Phase0 revoke audit: confirm smtp/wine/donors no longer get harmful A6."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_yaml  # noqa: E402
from src.policy_per import apply_per_config, clear_per_upgrades_cache  # noqa: E402


def main() -> int:
    clear_per_upgrades_cache()
    cfg = load_yaml(PROJECT_ROOT / "configs" / "adadae_per.yaml")
    checks = []
    ok = True
    for ds, expect_protect in [
        ("smtp", True),
        ("donors", True),
        ("wine", True),
        ("speech", False),
        ("glass", False),
        ("Wilt", False),
    ]:
        out = apply_per_config(
            cfg, "semi-supervised", "classical", ds, meta={"n": 1000.0, "d": 20.0}
        )
        ad = out["adadae"]
        pol = ad.get("resolved_policy", "")
        row = {
            "dataset": ds,
            "resolved_policy": pol,
            "use_apex": bool(ad.get("use_apex")),
            "use_nautilus": bool(ad.get("use_nautilus")),
            "use_mce": bool(ad.get("use_mce")),
            "protected": "protect" in pol,
        }
        if expect_protect:
            if ad.get("use_apex") or (ds == "wine" and ad.get("use_nautilus")):
                ok = False
                row["fail"] = "still has revoked A6"
            if "protect" not in pol and ds in ("smtp", "donors", "wine"):
                ok = False
                row["fail"] = "missing protect tag"
        checks.append(row)

    # wine must not have nautilus
    wine = next(c for c in checks if c["dataset"] == "wine")
    if wine["use_nautilus"]:
        ok = False

    report = {
        "pass": ok,
        "revoked": ["smtp:apex", "wine:nautilus", "speech:mce+smc+apex kitchen-sink"],
        "checks": checks,
    }
    out = PROJECT_ROOT / "results/adadae_per/thesis/phase0_revoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
