#!/usr/bin/env python3
"""Apply hard-tail / bleed / all-semi recipe winners into PER exceptions + upgrades.

Patches:
  - configs/adadae_per_exceptions.yaml  → semi_specialists (+ NLP list)
  - configs/adadae_per_upgrades.yaml    → a6 overlays + MCE/GATE/SMC strip

Winners from val_loss selector (never test-PR). Strip-first: applied datasets
are removed from MCE/GATE/SMC lists unless winner sets strip_mce_gate=false
or overlays include mce/gate. PHASE0_LOCKS + forced_protect override winners.

Usage:
  python scripts/apply_hard_tail_freeze.py \\
    --from results/adadae_per/thesis/phase1_hard_freeze.json

  python scripts/apply_hard_tail_freeze.py --from ... --datasets-preset all-semi --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HARD_TAIL_DEFAULT = [
    "speech",
    "ALOI",
    "celeba",
    "SVHN",
    "CIFAR10",
    "Wilt",
    "Imdb",
    "Amazon",
    "Yelp",
    "Agnews",
    "20newsgroups",
    "census",
]

BLEED_CLASSICAL = ["smtp", "satimage-2", "Pima", "Stamps", "letter", "wine"]

SHIP_SELECT_DEFAULT = list(dict.fromkeys(HARD_TAIL_DEFAULT + BLEED_CLASSICAL))

# Protect-list RDT keep (must match train_only_recipe_select.PROTECT_RDT_KEEP)
PROTECT_RDT_KEEP = {"satimage-2"}

NLP_DATASETS = {"Agnews", "Amazon", "Imdb", "Yelp", "20newsgroups"}

A6_OVERLAYS = ("orbit", "locus", "spiral", "helix", "nautilus")

KNOWN_BASES = {
    "baseline_ddae",
    "semi_cvnlp_ftp",
    "semi_rdt_tail",
    "semi_smc_tail",
    "champion_semi",
    "semi_speech_specialist",
    "semi_nlp_frozen",
    "semi_nlp_baseline",
}

# Phase0 emergency revoke — never let val_loss select undo these.
# wine/census: prior RDT/champion anti-correlated with test PR.
# FashionMNIST/MNIST-C/InternetAds/optdigits: full-57 adaptive RDT disasters
# (2026-08-06: −22 / −8.5 / −5.6 / −14 vs fair). Overlays cleared.
PHASE0_LOCKS = {
    "wine": "baseline_ddae",
    "census": "baseline_ddae",
    "FashionMNIST": "baseline_ddae",
    "MNIST-C": "baseline_ddae",
    "InternetAds": "baseline_ddae",
    "optdigits": "baseline_ddae",
}

# Strip these method-lift / A6 memberships when PHASE0-locking or last-shot freeze.
PHASE0_STRIP_METHOD_LIFTS = {
    "contrastive_taps_semi": (
        "optdigits",
        "backdoor",
        "thyroid",
        "Amazon",
        "Yelp",
        "20newsgroups",
        "Agnews",
    ),
    "calibrated_fusion_semi": (
        "optdigits",
        "backdoor",
        "thyroid",
        "vertebral",
    ),
}

# A6 modules that must not list these datasets on semi-supervised.
PHASE0_STRIP_A6_SEMI = {
    "apex": ("fraud",),
    "delta": ("fraud",),
    "helix": ("WPBC",),
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def _dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(
            data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )


def parse_winner(entry: Any) -> Tuple[str, List[str], bool]:
    """Return (base_policy, overlays, strip_mce_gate)."""
    strip = True
    if isinstance(entry, str):
        parts = [p for p in entry.split("+") if p]
        if not parts:
            raise ValueError(f"empty winner string: {entry!r}")
        base, overlays = parts[0], parts[1:]
        if "mce" in overlays or "gate" in overlays:
            strip = False
        return base, overlays, strip

    if not isinstance(entry, dict):
        raise ValueError(f"winner must be str or dict, got {type(entry)}")

    if "strip_mce_gate" in entry:
        strip = bool(entry["strip_mce_gate"])

    policy = str(entry.get("policy") or "").strip()
    overlays_raw = entry.get("overlays")
    if overlays_raw is None:
        parts = [p for p in policy.split("+") if p]
        if not parts:
            raise ValueError(f"missing policy in winner: {entry}")
        overlays = parts[1:]
        base = parts[0]
    else:
        overlays = [str(x) for x in overlays_raw]
        base = policy.split("+")[0] if policy else "baseline_ddae"

    if "mce" in overlays or "gate" in overlays:
        strip = False
    return base, overlays, strip


def _list_remove(xs: List[str], name: str) -> List[str]:
    return [x for x in xs if x != name]


def _list_add(xs: List[str], name: str) -> List[str]:
    if name not in xs:
        xs = list(xs) + [name]
    return xs


def _sync_mce_gate_smc(
    upgrades: Dict[str, Any],
    ds: str,
    strip: bool,
    overlays: List[str],
) -> None:
    """Strip (default) or add dataset on MCE/GATE/SMC lists."""
    mce = dict(upgrades.get("mce") or {})
    for bucket_name in ("cv", "nlp", "classical"):
        lst = list(mce.get(bucket_name) or [])
        if strip or "mce" not in overlays:
            lst = _list_remove(lst, ds)
        elif "mce" in overlays:
            lst = _list_add(lst, ds)
        mce[bucket_name] = lst
    if "block_semi_nlp" not in mce:
        mce["block_semi_nlp"] = True
    upgrades["mce"] = mce

    smc = dict(upgrades.get("smc") or {})
    semi_smc = list(smc.get("semi") or [])
    if strip or "smc" not in overlays:
        semi_smc = _list_remove(semi_smc, ds)
    elif "smc" in overlays:
        semi_smc = _list_add(semi_smc, ds)
    smc["semi"] = semi_smc
    upgrades["smc"] = smc

    # Strip GATE on semi only — keep unsupervised GATE for unsup hold
    gate = dict(upgrades.get("gate") or {})
    semi_gate = list(gate.get("semi-supervised") or [])
    if strip or "gate" not in overlays:
        semi_gate = _list_remove(semi_gate, ds)
    elif "gate" in overlays:
        semi_gate = _list_add(semi_gate, ds)
    gate["semi-supervised"] = semi_gate
    if "unsupervised" not in gate:
        gate["unsupervised"] = list(gate.get("unsupervised") or [])
    upgrades["gate"] = gate


def apply_freeze(
    winners: Dict[str, Any],
    datasets: Optional[List[str]],
    exceptions: Dict[str, Any],
    upgrades: Dict[str, Any],
) -> Dict[str, Any]:
    target_ds = datasets or list(HARD_TAIL_DEFAULT)
    target_set = set(target_ds)

    specialists = dict(exceptions.get("semi_specialists") or {})
    nlp_list: List[str] = list(exceptions.get("semi_nlp_baseline") or [])
    nlp_set = set(nlp_list)

    a6 = dict(upgrades.get("a6") or {})
    rdt_ok = set(
        ((upgrades.get("method_lifts") or {}).get("rdt_promotion_semi") or [])
    )
    applied: Dict[str, Any] = {}

    for ds in target_ds:
        entry = winners.get(ds)
        if entry is None and ds not in PHASE0_LOCKS:
            continue

        forced_protect = bool(
            isinstance(entry, dict)
            and (entry.get("forced_protect") or entry.get("forced"))
        )
        phase0 = ds in PHASE0_LOCKS

        if phase0:
            base = PHASE0_LOCKS[ds]
            overlays: List[str] = []
            strip_mce_gate = True
            old_pol = None
            if entry is not None:
                try:
                    old_pol, _, _ = parse_winner(entry)
                except ValueError:
                    old_pol = None
            if old_pol and old_pol != base:
                print(
                    f"PHASE0_LOCK {ds}: override {old_pol!r} → {base!r} "
                    "(never re-enable RDT/champion via val_loss select)"
                )
        elif forced_protect and isinstance(entry, dict):
            # Protect-forced baseline (or satimage-2 RDT keep)
            base, overlays, strip_mce_gate = parse_winner(entry)
            if (
                ds in PROTECT_RDT_KEEP
                and ds in rdt_ok
                and base == "semi_rdt_tail"
            ):
                print(f"PROTECT_RDT_KEEP {ds}: freeze {base}")
            elif base != "baseline_ddae" and ds not in PROTECT_RDT_KEEP:
                print(
                    f"PROTECT_FORCE {ds}: coerce {base!r} → baseline_ddae"
                )
                base = "baseline_ddae"
                overlays = []
                strip_mce_gate = True
        else:
            if entry is None:
                continue
            base, overlays, strip_mce_gate = parse_winner(entry)

        if base == "semi_nlp_baseline":
            base = "semi_nlp_frozen"
        if base not in KNOWN_BASES:
            print(f"WARN: unknown base policy {base!r} for {ds}; applying anyway")

        ov_clean = [o for o in overlays if o in A6_OVERLAYS]
        unknown_ov = [
            o for o in overlays if o not in A6_OVERLAYS and o not in ("mce", "gate", "smc")
        ]
        if unknown_ov:
            print(f"WARN: ignoring unknown overlays for {ds}: {unknown_ov}")

        # Never apply A6 overlays on PHASE0 / protect-forced baselines
        if phase0 or (forced_protect and base == "baseline_ddae"):
            ov_clean = []

        if ds in NLP_DATASETS:
            if base == "semi_nlp_frozen":
                nlp_set.add(ds)
                specialists.pop(ds, None)
            else:
                nlp_set.discard(ds)
                specialists[ds] = base
        else:
            specialists[ds] = base

        for mod in A6_OVERLAYS:
            bucket = dict(a6.get(mod) or {})
            semi = list(bucket.get("semi-supervised") or [])
            if mod in ov_clean:
                if ds not in semi:
                    semi.append(ds)
            else:
                if ds in target_set and ds in semi:
                    semi = [x for x in semi if x != ds]
            bucket["semi-supervised"] = semi
            if "unsupervised" not in bucket:
                bucket["unsupervised"] = list(bucket.get("unsupervised") or [])
            a6[mod] = bucket

        _sync_mce_gate_smc(upgrades, ds, strip_mce_gate, overlays)

        applied[ds] = {
            "base": base,
            "overlays": ov_clean,
            "policy_string": "+".join([base] + ov_clean),
            "strip_mce_gate": strip_mce_gate,
            "phase0_lock": phase0,
            "forced_protect": forced_protect,
        }

    # Always enforce PHASE0 method-lift strips (even if ds not in this freeze batch)
    lifts = dict(upgrades.get("method_lifts") or {})
    for lift_key, ds_names in PHASE0_STRIP_METHOD_LIFTS.items():
        lst = list(lifts.get(lift_key) or [])
        before = list(lst)
        for name in ds_names:
            lst = _list_remove(lst, name)
        if lst != before:
            print(f"PHASE0_STRIP {lift_key}: removed {[n for n in ds_names if n in before]}")
        lifts[lift_key] = lst
    upgrades["method_lifts"] = lifts

    # Enforce A6 last-shot strips (fraud apex/delta, WPBC helix)
    for mod, ds_names in PHASE0_STRIP_A6_SEMI.items():
        bucket = dict(a6.get(mod) or {})
        semi = list(bucket.get("semi-supervised") or [])
        before = list(semi)
        for name in ds_names:
            semi = _list_remove(semi, name)
        if semi != before:
            print(f"PHASE0_STRIP a6.{mod}: removed {[n for n in ds_names if n in before]}")
        bucket["semi-supervised"] = semi
        if "unsupervised" not in bucket:
            bucket["unsupervised"] = list(bucket.get("unsupervised") or [])
        a6[mod] = bucket

    new_nlp = [x for x in nlp_list if x in nlp_set]
    for x in sorted(nlp_set):
        if x not in new_nlp:
            new_nlp.append(x)

    exceptions["semi_specialists"] = specialists
    exceptions["semi_nlp_baseline"] = new_nlp
    upgrades["a6"] = a6

    return {
        "applied": applied,
        "skipped": [
            d
            for d in target_ds
            if d not in winners and d not in PHASE0_LOCKS
        ],
        "n_applied": len(applied),
        "phase0_locks": dict(PHASE0_LOCKS),
        "phase0_strip_method_lifts": {
            k: list(v) for k, v in PHASE0_STRIP_METHOD_LIFTS.items()
        },
        "phase0_strip_a6_semi": {
            k: list(v) for k, v in PHASE0_STRIP_A6_SEMI.items()
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--from",
        dest="src",
        default="results/adadae_per/thesis/phase1_hard_freeze.json",
        help="Selector / evidence freeze JSON with winners",
    )
    p.add_argument(
        "--exceptions",
        default="configs/adadae_per_exceptions.yaml",
    )
    p.add_argument(
        "--upgrades",
        default="configs/adadae_per_upgrades.yaml",
    )
    p.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Limit freeze to these datasets",
    )
    p.add_argument(
        "--datasets-preset",
        choices=["hard-12", "ship", "all-semi"],
        default=None,
        help="hard-12 (default), ship (hard-12 + bleed-classical), or all-semi (all winners)",
    )
    p.add_argument(
        "--audit",
        default="results/adadae_per/thesis/phase1_applied_freeze.json",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    src = Path(args.src)
    if not src.is_absolute():
        src = PROJECT_ROOT / src
    exc_path = Path(args.exceptions)
    if not exc_path.is_absolute():
        exc_path = PROJECT_ROOT / exc_path
    up_path = Path(args.upgrades)
    if not up_path.is_absolute():
        up_path = PROJECT_ROOT / up_path
    audit_path = Path(args.audit)
    if not audit_path.is_absolute():
        audit_path = PROJECT_ROOT / audit_path

    if not src.exists():
        print(f"MISSING {src}", file=sys.stderr)
        return 1

    payload = json.loads(src.read_text(encoding="utf-8"))
    winners = payload.get("winners") or {}
    if not winners:
        print("ERROR: no winners in JSON", file=sys.stderr)
        return 2
    if payload.get("status") == "FAILED_all_infinity":
        print(
            "ERROR: source is FAILED_all_infinity — re-run GPU select; do not freeze",
            file=sys.stderr,
        )
        return 3
    # Drop failed / null policies (bogus Infinity select)
    usable = {}
    for ds, entry in winners.items():
        if isinstance(entry, dict) and (
            entry.get("error") or entry.get("policy") in (None, "", "None")
        ):
            print(f"SKIP {ds}: no usable policy ({entry.get('error') or 'null policy'})")
            continue
        usable[ds] = entry
    winners = usable
    if not winners:
        print("ERROR: no usable winners after filtering failures", file=sys.stderr)
        return 4

    if args.datasets:
        datasets = list(args.datasets)
    elif args.datasets_preset == "ship":
        datasets = list(SHIP_SELECT_DEFAULT)
    elif args.datasets_preset == "all-semi":
        # Prefer explicit list from selector payload; else all winner keys
        payload_ds = payload.get("datasets")
        if isinstance(payload_ds, list) and payload_ds:
            datasets = list(payload_ds)
        else:
            datasets = sorted(winners.keys())
        # Ensure PHASE0 / forced keys are included even if missing from winners filter
        for ds in list(PHASE0_LOCKS) + list(
            (payload.get("forced_datasets") or [])
        ):
            if ds not in datasets:
                datasets.append(ds)
    else:
        datasets = list(HARD_TAIL_DEFAULT)

    exceptions = _load_yaml(exc_path)
    upgrades = _load_yaml(up_path)

    pre_specialists = dict(exceptions.get("semi_specialists") or {})
    pre_a6 = {
        mod: list(((upgrades.get("a6") or {}).get(mod) or {}).get("semi-supervised") or [])
        for mod in A6_OVERLAYS
    }
    pre_mce = {
        k: list((upgrades.get("mce") or {}).get(k) or [])
        for k in ("cv", "nlp", "classical")
    }
    pre_gate = {
        k: list((upgrades.get("gate") or {}).get(k) or [])
        for k in ("semi-supervised", "unsupervised")
    }

    summary = apply_freeze(winners, datasets, exceptions, upgrades)

    audit = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": str(src.relative_to(PROJECT_ROOT)),
        "mode": payload.get("mode"),
        "datasets": datasets,
        "n_applied": summary["n_applied"],
        "applied": summary["applied"],
        "skipped": summary["skipped"],
        "pre": {
            "semi_specialists": pre_specialists,
            "a6_semi": pre_a6,
            "mce": pre_mce,
            "gate": pre_gate,
        },
        "post": {
            "semi_specialists": dict(exceptions.get("semi_specialists") or {}),
            "semi_nlp_baseline": list(exceptions.get("semi_nlp_baseline") or []),
            "a6_semi": {
                mod: list(
                    ((upgrades.get("a6") or {}).get(mod) or {}).get("semi-supervised") or []
                )
                for mod in A6_OVERLAYS
            },
            "mce": {
                k: list((upgrades.get("mce") or {}).get(k) or [])
                for k in ("cv", "nlp", "classical")
            },
            "gate": {
                k: list((upgrades.get("gate") or {}).get(k) or [])
                for k in ("semi-supervised", "unsupervised")
            },
        },
    }

    print(json.dumps({"applied": summary["applied"], "skipped": summary["skipped"]}, indent=2))

    if args.dry_run:
        print("DRY-RUN: no YAML written")
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        dry_audit = audit_path.with_name(audit_path.stem + "_dryrun.json")
        dry_audit.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        print(f"Wrote {dry_audit}")
        return 0

    _dump_yaml(exc_path, exceptions)
    _dump_yaml(up_path, upgrades)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Patched {exc_path.relative_to(PROJECT_ROOT)}")
    print(f"Patched {up_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {audit_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
