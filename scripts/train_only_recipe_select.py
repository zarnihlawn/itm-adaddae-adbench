#!/usr/bin/env python3
"""Train-only multi-recipe arbitration for hard semi datasets (beat-paper Phase 1).

Picks winners by **val_loss** (never test PR). Supports base policies plus
A6 overlays (orbit/locus/spiral/helix). Can emit a freeze YAML patch.

Usage:
  python scripts/train_only_recipe_select.py --dry-run
  python scripts/train_only_recipe_select.py --datasets speech Wilt CIFAR10 --seeds 111 222 --hardware 16gb
  python scripts/train_only_recipe_select.py --apply-evidence-freeze  # no GPU: write evidence freeze JSON
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

HARD_SEMI_DEFAULT = [
    "speech",
    "celeba",
    "Imdb",
    "ALOI",
    "Amazon",
    "Wilt",
    "SVHN",
    "Yelp",
    "20newsgroups",
    "CIFAR10",
    "census",
    "Agnews",
    "vertebral",
    "glass",
    "WPBC",
]

# Base policy ids in POLICY_REGISTRY
BASE_CANDIDATES = [
    "baseline_ddae",
    "semi_cvnlp_ftp",
    "semi_rdt_tail",
    "semi_smc_tail",
    "champion_semi",
    "semi_speech_specialist",
]

# Overlay flags applied on top of a base (static)
OVERLAY_CANDIDATES: Dict[str, Dict[str, Any]] = {
    "orbit": {"adadae": {"use_orbit": True, "use_multiview": True}},
    "locus": {"adadae": {"use_locus": True, "use_multiview": True}},
    "spiral": {"adadae": {"use_spiral": True}},
    "helix": {"adadae": {"use_helix": True}},
}

# Evidence freeze when GPU selection cannot run (from winloss + ADBench analysis)
EVIDENCE_FREEZE: Dict[str, Dict[str, Any]] = {
    "speech": {"policy": "semi_rdt_tail", "overlays": ["locus", "spiral", "helix"], "reason": "kitchen-sink failed; RDT designed for speech"},
    "ALOI": {"policy": "semi_rdt_tail", "overlays": ["orbit", "locus"], "reason": "RDT+orbit for CV embeds"},
    "celeba": {"policy": "semi_cvnlp_ftp", "overlays": ["orbit"], "reason": "FTP+orbit; drop nlp_baseline misroute"},
    "CIFAR10": {"policy": "semi_cvnlp_ftp", "overlays": ["orbit", "helix"], "reason": "FTP+orbit/helix"},
    "SVHN": {"policy": "semi_cvnlp_ftp", "overlays": ["orbit", "helix"], "reason": "FTP+orbit"},
    "Wilt": {"policy": "semi_rdt_tail", "overlays": ["orbit", "locus", "spiral"], "reason": "robust RDT+extra views"},
    "census": {"policy": "baseline_ddae", "overlays": ["locus"], "reason": "baseline+locus"},
    "vertebral": {"policy": "baseline_ddae", "overlays": ["nautilus"], "reason": "keep nautilus win vs fair"},
    "glass": {"policy": "baseline_ddae", "overlays": ["nautilus"], "reason": "keep nautilus win vs fair"},
    "WPBC": {"policy": "semi_speech_specialist", "overlays": [], "reason": "existing specialist"},
    "Imdb": {"policy": "semi_nlp_frozen", "overlays": ["orbit"], "reason": "NLP frozen+unit_norm/orbit"},
    "Amazon": {"policy": "semi_nlp_frozen", "overlays": ["orbit"], "reason": "NLP frozen+orbit"},
    "Yelp": {"policy": "semi_nlp_frozen", "overlays": ["orbit"], "reason": "NLP frozen+orbit"},
    "Agnews": {"policy": "semi_nlp_frozen", "overlays": ["orbit"], "reason": "NLP frozen+orbit"},
    "20newsgroups": {"policy": "semi_nlp_frozen", "overlays": ["orbit"], "reason": "NLP frozen+orbit"},
}


def _category_for(name: str) -> str:
    cv = {"CIFAR10", "SVHN", "MNIST-C", "FashionMNIST", "MVTec-AD", "celeba", "ALOI", "speech"}
    nlp = {"Agnews", "Amazon", "Imdb", "Yelp", "20newsgroups"}
    if name in cv:
        return "cv"
    if name in nlp:
        return "nlp"
    return "classical"


def dry_run_resolve() -> Dict[str, Any]:
    from src.config import load_yaml
    from src.policy_per import apply_per_config, clear_per_upgrades_cache

    clear_per_upgrades_cache()
    cfg = load_yaml(PROJECT_ROOT / "configs" / "adadae_per.yaml")
    rows = []
    for ds in HARD_SEMI_DEFAULT:
        cat = _category_for(ds)
        out = apply_per_config(
            cfg, "semi-supervised", cat, ds, meta={"n": 800.0, "d": 32.0}
        )
        rows.append(
            {
                "dataset": ds,
                "category": cat,
                "resolved_policy": out["adadae"].get("resolved_policy"),
                "use_scs": out["adadae"].get("use_scs"),
                "use_mce": out["adadae"].get("use_mce"),
                "use_orbit": out["adadae"].get("use_orbit"),
                "use_locus": out["adadae"].get("use_locus"),
                "use_nautilus": out["adadae"].get("use_nautilus"),
                "use_apex": out["adadae"].get("use_apex"),
                "fusion_mode": out["adadae"].get("fusion_mode"),
                "features": out.get("features"),
                "train_ov": {
                    k: out.get("train", {}).get(k)
                    for k in ("min_epochs", "val_fraction_semi", "early_stop_patience")
                    if out.get("train", {}).get(k) is not None
                },
            }
        )
    return {"mode": "dry_run", "base_candidates": BASE_CANDIDATES, "current_per": rows}


def evidence_freeze_report() -> Dict[str, Any]:
    return {
        "mode": "evidence_freeze",
        "note": "GPU selector unavailable locally; freeze from winloss+ADBench analysis. Re-run selector on Vast to refine.",
        "winners": EVIDENCE_FREEZE,
        "protect_baseline_semi": [
            "smtp", "donors", "wine", "http", "musk", "breastw", "shuttle",
            "Ionosphere", "Lymphography", "pendigits", "magic.gamma", "satimage-2",
            "vowels", "skin", "PageBlocks",
        ],
        "revoked": {
            "smtp": "apex (−10.3 vs fair)",
            "wine": "nautilus (−5.0 vs fair)",
            "speech": "mce+smc+apex+orbit+delta kitchen-sink",
        },
    }


def _candidate_names() -> List[str]:
    names = list(BASE_CANDIDATES)
    for ov in OVERLAY_CANDIDATES:
        names.append(f"baseline_ddae+{ov}")
        names.append(f"semi_cvnlp_ftp+{ov}")
        names.append(f"semi_rdt_tail+{ov}")
    return names


def _build_candidate_cfg(base_cfg: Dict[str, Any], cand: str) -> Dict[str, Any]:
    from src.policy import policy_overrides, _deep_update

    parts = cand.split("+")
    base = parts[0]
    cfg = copy.deepcopy(base_cfg)
    cfg["adadae"] = dict(cfg.get("adadae") or {})
    cfg["adadae"]["policy"] = "static"
    cfg["adadae"].pop("exceptions_file", None)
    cfg["adadae"].pop("upgrades_file", None)
    cfg = _deep_update(cfg, policy_overrides(base))
    for ov in parts[1:]:
        if ov in OVERLAY_CANDIDATES:
            cfg = _deep_update(cfg, OVERLAY_CANDIDATES[ov])
        elif ov == "nautilus":
            cfg.setdefault("adadae", {})["use_nautilus"] = True
    cfg["adadae"]["resolved_policy"] = f"select:{cand}"
    cfg["paths"] = dict(cfg["paths"])
    cfg["paths"]["results_dir"] = "results/adadae_per_select"
    cfg["paths"]["run_id"] = "adadae_per_select"
    return cfg


def run_selection(
    datasets: List[str],
    seeds: List[int],
    hardware: Optional[str],
    candidates: Optional[List[str]] = None,
) -> Dict[str, Any]:
    from src.config import load_config
    from src.data.datasets import build_registry
    from src.train.experiment import run_single_file, set_seed

    base = load_config(str(PROJECT_ROOT / "configs" / "adadae_per.yaml"))
    if hardware:
        hw = hardware if hardware.endswith(".yaml") else f"hardware_{hardware}.yaml"
        # detect_hardware style keys: 16gb -> hardware_rtx5070ti via --hardware in protocol
        base["hardware"] = hw if "hardware_" in hw else f"hardware_{hardware}.yaml"

    cands = candidates or [
        "baseline_ddae",
        "semi_cvnlp_ftp",
        "semi_rdt_tail",
        "champion_semi",
        "baseline_ddae+orbit",
        "semi_rdt_tail+orbit",
        "semi_rdt_tail+locus",
        "semi_cvnlp_ftp+orbit",
        "semi_cvnlp_ftp+helix",
    ]

    registry = build_registry(base["paths"]["adbench_root"])
    by_name = {e.name: e for e in registry}
    results: Dict[str, Any] = {"winners": {}, "jobs": [], "candidates": cands}

    for ds in datasets:
        if ds not in by_name:
            print(f"SKIP missing dataset {ds}")
            continue
        entry = by_name[ds]
        cat = entry.category
        ds_votes: Dict[str, List[float]] = {c: [] for c in cands}
        for seed in seeds:
            best_name = None
            best_val = float("inf")
            for cand in cands:
                cfg = _build_candidate_cfg(base, cand)
                set_seed(seed)
                try:
                    out = run_single_file(
                        entry.path,
                        entry.name,
                        entry.split,
                        "semi-supervised",
                        seed,
                        cfg,
                        category=cat,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"FAIL {ds} {cand} seed={seed}: {exc}")
                    continue
                val = float(out.get("best_val_metric", float("inf")))
                ds_votes[cand].append(val)
                results["jobs"].append(
                    {
                        "dataset": ds,
                        "seed": seed,
                        "candidate": cand,
                        "val_loss": val,
                        "PR": (out.get("metrics") or {}).get("PR-AUC"),
                    }
                )
                if val < best_val:
                    best_val = val
                    best_name = cand
            if best_name:
                results.setdefault("_seed_winners", {}).setdefault(ds, []).append(best_name)

        means = {c: (sum(v) / len(v) if v else float("inf")) for c, v in ds_votes.items()}
        winner = min(means, key=means.get) if means else "baseline_ddae"
        results["winners"][ds] = {"policy": winner, "mean_val_loss": means}
    return results


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="*", default=HARD_SEMI_DEFAULT)
    p.add_argument("--seeds", nargs="*", type=int, default=[111, 222, 333])
    p.add_argument("--hardware", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--apply-evidence-freeze", action="store_true")
    args = p.parse_args()

    if args.dry_run:
        report = dry_run_resolve()
        out = PROJECT_ROOT / "results/adadae_per/thesis/loop6_train_only_select.json"
    elif args.apply_evidence_freeze:
        report = evidence_freeze_report()
        out = PROJECT_ROOT / "results/adadae_per/thesis/phase1_hard_freeze.json"
    else:
        report = run_selection(args.datasets, args.seeds, args.hardware)
        out = PROJECT_ROOT / "results/adadae_per/thesis/phase1_hard_freeze.json"

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    printable = {k: report[k] for k in report if k != "jobs"}
    print(json.dumps(printable, indent=2)[:4000])
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
