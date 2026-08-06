#!/usr/bin/env python3
"""Phase0 / last-shot revoke audit: locks, taps/cal_fuse/A6 strips."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.apply_hard_tail_freeze import (  # noqa: E402
    PHASE0_LOCKS,
    PHASE0_STRIP_A6_SEMI,
    PHASE0_STRIP_METHOD_LIFTS,
)
from src.config import load_yaml  # noqa: E402
from src.policy_per import apply_per_config, clear_per_upgrades_cache  # noqa: E402


def _category(ds: str) -> str:
    if ds in ("Agnews", "Amazon", "Imdb", "Yelp", "20newsgroups"):
        return "nlp"
    if ds in (
        "FashionMNIST",
        "MNIST-C",
        "CIFAR10",
        "SVHN",
        "celeba",
        "ALOI",
        "mnist",
        "MVTec-AD",
    ):
        return "cv"
    return "classical"


def main() -> int:
    clear_per_upgrades_cache()
    cfg = load_yaml(PROJECT_ROOT / "configs" / "adadae_per.yaml")
    upgrades = load_yaml(PROJECT_ROOT / "configs" / "adadae_per_upgrades.yaml")
    lifts = upgrades.get("method_lifts") or {}
    a6 = upgrades.get("a6") or {}
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
        ("FashionMNIST", False),
        ("MNIST-C", False),
        ("InternetAds", False),
        ("optdigits", False),
        ("backdoor", False),
        ("thyroid", False),
        ("vertebral", False),
        ("Amazon", False),
        ("Yelp", False),
        ("20newsgroups", False),
        ("Imdb", False),
        ("fraud", False),
        ("WPBC", False),
        ("cover", False),
        ("Hepatitis", False),
    ]:
        cat = _category(ds)
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
            "use_delta": bool(ad.get("use_delta")),
            "use_helix": bool(ad.get("use_helix")),
            "use_mce": bool(ad.get("use_mce")),
            "use_rejection_training": bool(ad.get("use_rejection_training")),
            "use_ftp": bool(ad.get("use_ftp")),
            "contrastive": bool((out.get("train") or {}).get("contrastive")),
            "cal_fuse": ad.get("fusion_mode") == "calibrated",
            "protected": "protect" in pol,
        }
        if expect_protect:
            if ad.get("use_apex") or (ds == "wine" and ad.get("use_nautilus")):
                ok = False
                row["fail"] = "still has revoked A6"
            if "protect" not in pol and ds in ("smtp", "donors", "wine", "census"):
                ok = False
                row["fail"] = "missing protect tag"
        if ds in PHASE0_LOCKS:
            if "baseline_ddae" not in pol:
                ok = False
                row["fail"] = f"PHASE0_LOCK: expected baseline_ddae, got {pol}"
            if ad.get("use_rejection_training"):
                ok = False
                row["fail"] = "PHASE0_LOCK: RDT still on"
            if ad.get("use_ftp") and ds == "wine":
                ok = False
                row["fail"] = "PHASE0_LOCK: FTP still on (champion leak)"
        if ds in (
            "Agnews",
            "Amazon",
            "Yelp",
            "20newsgroups",
            "backdoor",
            "thyroid",
            "optdigits",
        ) and (out.get("train") or {}).get("contrastive"):
            ok = False
            row["fail"] = f"{ds}: TAPS still on after last-shot strip"
        if ds == "Imdb" and not (out.get("train") or {}).get("contrastive"):
            ok = False
            row["fail"] = "Imdb: expected taps kept (only proven NLP taps)"
        if ds in ("backdoor", "optdigits", "thyroid", "vertebral") and ad.get(
            "fusion_mode"
        ) == "calibrated":
            ok = False
            row["fail"] = f"{ds}: cal_fuse still on after last-shot strip"
        if ds == "fraud" and (ad.get("use_apex") or ad.get("use_delta")):
            ok = False
            row["fail"] = "fraud: apex/delta still on"
        if ds == "WPBC" and ad.get("use_helix"):
            ok = False
            row["fail"] = "WPBC: helix still on"
        if ds == "cover" and not (ad.get("use_apex") and ad.get("use_delta")):
            ok = False
            row["fail"] = "cover: expected apex+delta kept"
        if ds == "Hepatitis" and not (
            ad.get("use_rejection_training")
            and ad.get("fusion_mode") == "calibrated"
        ):
            ok = False
            row["fail"] = "Hepatitis: expected RDT+cal_fuse kept"
        checks.append(row)

    for lift_key, ds_names in PHASE0_STRIP_METHOD_LIFTS.items():
        lst = set(lifts.get(lift_key) or [])
        leaked = [n for n in ds_names if n in lst]
        if leaked:
            ok = False
            checks.append(
                {
                    "dataset": f"_lift:{lift_key}",
                    "fail": f"still lists {leaked}",
                    "resolved_policy": str(sorted(lst)),
                }
            )

    for mod, ds_names in PHASE0_STRIP_A6_SEMI.items():
        semi = set((a6.get(mod) or {}).get("semi-supervised") or [])
        leaked = [n for n in ds_names if n in semi]
        if leaked:
            ok = False
            checks.append(
                {
                    "dataset": f"_a6:{mod}",
                    "fail": f"still lists {leaked}",
                    "resolved_policy": str(sorted(semi)),
                }
            )

    report = {
        "pass": ok,
        "revoked": [
            "smtp:apex",
            "wine/census:RDT→baseline_ddae",
            "FashionMNIST/MNIST-C/InternetAds/optdigits:RDT→baseline_ddae",
            "Agnews/Amazon/Yelp/20newsgroups:taps stripped (Imdb taps kept)",
            "backdoor/thyroid/optdigits/vertebral:cal_fuse stripped",
            "fraud:apex+delta stripped",
            "WPBC:helix stripped",
            "keep: Hepatitis RDT+cal_fuse, cover apex+delta+cal_fuse",
        ],
        "phase0_locks": dict(PHASE0_LOCKS),
        "phase0_strip_method_lifts": {
            k: list(v) for k, v in PHASE0_STRIP_METHOD_LIFTS.items()
        },
        "phase0_strip_a6_semi": {
            k: list(v) for k, v in PHASE0_STRIP_A6_SEMI.items()
        },
        "checks": checks,
    }
    out_path = PROJECT_ROOT / "results/adadae_per/thesis/phase0_revoke.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
