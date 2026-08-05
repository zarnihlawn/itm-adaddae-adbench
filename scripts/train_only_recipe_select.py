#!/usr/bin/env python3
"""Train-only multi-recipe arbitration (val_loss only — never test PR).

Strip-first ablations for bleed-CV (plain bases + at most one A6 overlay).
Presets: ship (hard-12 + bleed-classical), bleed-cv, bleed-classical, hard-12.

Usage:
  python scripts/train_only_recipe_select.py --dry-run
  python scripts/train_only_recipe_select.py --preset ship --seeds 111 222 333 --hardware 16gb
  python scripts/train_only_recipe_select.py --apply-evidence-freeze
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Ship hard-tail set (12) — matches invalidate --hard-tails / apply freeze default
HARD_TAIL_12 = [
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

# Worst CV / embed stacks (kitchen-sink ablate-first)
BLEED_CV = ["SVHN", "ALOI", "celeba", "CIFAR10", "speech"]

# Protect / classical losers outside hard-12
BLEED_CLASSICAL = ["smtp", "satimage-2", "Pima", "Stamps", "letter", "wine"]

# Default GPU select = hard-12 ∪ bleed-classical (unique, hard-12 order first)
SHIP_SELECT_DEFAULT: List[str] = list(
    dict.fromkeys(HARD_TAIL_12 + BLEED_CLASSICAL)
)

# Legacy dry-run list (includes glass/vertebral/WPBC for routing dump)
HARD_SEMI_DEFAULT = HARD_TAIL_12 + ["vertebral", "glass", "WPBC"]

# ADBench NLP names
_NLP = {"Agnews", "Amazon", "Imdb", "Yelp", "20newsgroups"}
# True CV embeds in ADBench CV_by_ResNet18 (NOT Classical speech/ALOI/celeba)
_CV = {"CIFAR10", "SVHN", "MNIST-C", "FashionMNIST", "MVTec-AD"}

BASE_CANDIDATES = [
    "baseline_ddae",
    "semi_cvnlp_ftp",
    "semi_rdt_tail",
    "semi_smc_tail",
    "champion_semi",
    "semi_speech_specialist",
]

# Single-overlay only (strip-first; no multi-A6 kitchen-sink in search)
OVERLAY_CANDIDATES: Dict[str, Dict[str, Any]] = {
    "orbit": {"adadae": {"use_orbit": True, "use_multiview": True}},
    "locus": {"adadae": {"use_locus": True, "use_multiview": True}},
    "helix": {"adadae": {"use_helix": True}},
}

# Default search: plain bases + base+one overlay (no MCE/GATE — static cfg)
STRIP_FIRST_CANDIDATES = [
    "baseline_ddae",
    "semi_cvnlp_ftp",
    "semi_rdt_tail",
    "champion_semi",
    "baseline_ddae+orbit",
    "baseline_ddae+locus",
    "baseline_ddae+helix",
    "semi_rdt_tail+orbit",
    "semi_rdt_tail+locus",
    "semi_rdt_tail+helix",
    "semi_cvnlp_ftp+orbit",
    "semi_cvnlp_ftp+locus",
    "semi_cvnlp_ftp+helix",
]

# Evidence freeze when GPU selection cannot run (ablate-first revision for CV)
EVIDENCE_FREEZE: Dict[str, Dict[str, Any]] = {
    "speech": {
        "policy": "semi_rdt_tail",
        "overlays": ["locus"],
        "strip_mce_gate": True,
        "reason": "strip kitchen-sink; RDT+single locus",
    },
    "ALOI": {
        "policy": "semi_rdt_tail",
        "overlays": ["orbit"],
        "strip_mce_gate": True,
        "reason": "strip MCE+GATE multi-A6; RDT+orbit only",
    },
    "celeba": {
        "policy": "semi_cvnlp_ftp",
        "overlays": ["orbit"],
        "strip_mce_gate": True,
        "reason": "FTP+orbit; strip MCE",
    },
    "CIFAR10": {
        "policy": "semi_cvnlp_ftp",
        "overlays": ["orbit"],
        "strip_mce_gate": True,
        "reason": "FTP+orbit; strip GATE/helix stack",
    },
    "SVHN": {
        "policy": "semi_cvnlp_ftp",
        "overlays": [],
        "strip_mce_gate": True,
        "reason": "worst vs fair — strip MCE+orbit+helix to plain FTP",
    },
    "Wilt": {
        "policy": "semi_rdt_tail",
        "overlays": ["orbit", "locus", "spiral"],
        "reason": "keep winning RDT stack",
    },
    "census": {"policy": "baseline_ddae", "overlays": ["locus"], "reason": "baseline+locus"},
    "vertebral": {
        "policy": "baseline_ddae",
        "overlays": ["nautilus"],
        "reason": "keep nautilus win vs fair",
    },
    "glass": {
        "policy": "baseline_ddae",
        "overlays": ["nautilus"],
        "reason": "keep nautilus win vs fair",
    },
    "WPBC": {"policy": "semi_speech_specialist", "overlays": [], "reason": "existing specialist"},
    "Imdb": {"policy": "semi_nlp_frozen", "overlays": ["orbit"], "reason": "NLP frozen+orbit"},
    "Amazon": {"policy": "semi_nlp_frozen", "overlays": ["orbit"], "reason": "NLP frozen+orbit"},
    "Yelp": {"policy": "semi_nlp_frozen", "overlays": ["orbit"], "reason": "NLP frozen+orbit"},
    "Agnews": {"policy": "semi_nlp_frozen", "overlays": ["orbit"], "reason": "NLP frozen+orbit"},
    "20newsgroups": {
        "policy": "semi_nlp_frozen",
        "overlays": ["orbit"],
        "reason": "NLP frozen+orbit",
    },
    # Bleed-classical: force baseline (protect-compatible); strip upgrades via freeze
    "smtp": {
        "policy": "baseline_ddae",
        "overlays": [],
        "strip_mce_gate": True,
        "reason": "protect loser — baseline only",
    },
    "satimage-2": {
        "policy": "baseline_ddae",
        "overlays": [],
        "strip_mce_gate": True,
        "reason": "protect loser — baseline only",
    },
    "wine": {
        "policy": "baseline_ddae",
        "overlays": [],
        "strip_mce_gate": True,
        "reason": "protect loser — baseline only",
    },
    "Pima": {
        "policy": "baseline_ddae",
        "overlays": [],
        "strip_mce_gate": True,
        "reason": "bleed-classical loser",
    },
    "Stamps": {
        "policy": "baseline_ddae",
        "overlays": [],
        "strip_mce_gate": True,
        "reason": "bleed-classical loser",
    },
    "letter": {
        "policy": "baseline_ddae",
        "overlays": [],
        "strip_mce_gate": True,
        "reason": "bleed-classical loser",
    },
}


def _category_for(name: str, registry_by_name: Optional[Dict[str, Any]] = None) -> str:
    """Match ADBench registry categories (speech/ALOI/celeba are Classical)."""
    if registry_by_name and name in registry_by_name:
        return str(registry_by_name[name].category)
    if name in _NLP:
        return "nlp"
    if name in _CV:
        return "cv"
    return "classical"


def _preset_datasets(preset: str) -> List[str]:
    if preset == "ship":
        return list(SHIP_SELECT_DEFAULT)
    if preset == "hard-12":
        return list(HARD_TAIL_12)
    if preset == "bleed-cv":
        return list(BLEED_CV)
    if preset == "bleed-classical":
        return list(BLEED_CLASSICAL)
    if preset == "legacy":
        return list(HARD_SEMI_DEFAULT)
    raise ValueError(f"unknown preset {preset!r}")


def dry_run_resolve() -> Dict[str, Any]:
    from src.config import load_yaml
    from src.data.datasets import build_registry
    from src.policy_per import apply_per_config, clear_per_upgrades_cache

    clear_per_upgrades_cache()
    cfg = load_yaml(PROJECT_ROOT / "configs" / "adadae_per.yaml")
    registry_by_name: Dict[str, Any] = {}
    try:
        root = cfg.get("paths", {}).get("adbench_root")
        if root:
            registry_by_name = {e.name: e for e in build_registry(root)}
    except Exception:  # noqa: BLE001
        registry_by_name = {}

    rows = []
    for ds in HARD_SEMI_DEFAULT:
        cat = _category_for(ds, registry_by_name)
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
    return {
        "mode": "dry_run",
        "base_candidates": BASE_CANDIDATES,
        "strip_first_candidates": STRIP_FIRST_CANDIDATES,
        "presets": {
            "ship": SHIP_SELECT_DEFAULT,
            "hard-12": HARD_TAIL_12,
            "bleed-cv": BLEED_CV,
            "bleed-classical": BLEED_CLASSICAL,
        },
        "current_per": rows,
    }


def evidence_freeze_report() -> Dict[str, Any]:
    return {
        "mode": "evidence_freeze",
        "note": (
            "Ablate-first evidence freeze (CV strip MCE/GATE). "
            "Re-run --preset ship on Vast GPU for val_loss winners."
        ),
        "winners": EVIDENCE_FREEZE,
        "protect_baseline_semi": [
            "smtp",
            "donors",
            "wine",
            "http",
            "musk",
            "breastw",
            "shuttle",
            "Ionosphere",
            "Lymphography",
            "pendigits",
            "magic.gamma",
            "satimage-2",
            "vowels",
            "skin",
            "PageBlocks",
        ],
        "revoked": {
            "smtp": "apex (−10.3 vs fair)",
            "wine": "nautilus (−5.0 vs fair)",
            "speech": "mce+smc+apex+orbit+delta kitchen-sink",
            "SVHN": "mce+orbit+helix kitchen-sink (−4.57 vs fair)",
            "ALOI": "mce+gate+orbit+locus kitchen-sink (−2.98 vs fair)",
        },
        "sets": {
            "hard_tail_12": HARD_TAIL_12,
            "bleed_cv": BLEED_CV,
            "bleed_classical": BLEED_CLASSICAL,
            "ship_select": SHIP_SELECT_DEFAULT,
        },
    }


def _build_candidate_cfg(base_cfg: Dict[str, Any], cand: str) -> Dict[str, Any]:
    from src.policy import policy_overrides, _deep_update

    parts = cand.split("+")
    base = parts[0]
    cfg = copy.deepcopy(base_cfg)
    cfg["adadae"] = dict(cfg.get("adadae") or {})
    cfg["adadae"]["policy"] = "static"
    cfg["adadae"].pop("exceptions_file", None)
    cfg["adadae"].pop("upgrades_file", None)
    # Explicit strip: no MCE/GATE/SMC from PER upgrades (static)
    cfg["adadae"]["use_mce"] = False
    cfg["adadae"]["use_gate"] = False
    cfg = _deep_update(cfg, policy_overrides(base))
    cfg.setdefault("adadae", {})["use_mce"] = False
    cfg["adadae"]["use_gate"] = False
    for ov in parts[1:]:
        if ov in OVERLAY_CANDIDATES:
            cfg = _deep_update(cfg, OVERLAY_CANDIDATES[ov])
        elif ov == "nautilus":
            cfg.setdefault("adadae", {})["use_nautilus"] = True
        elif ov == "spiral":
            cfg.setdefault("adadae", {})["use_spiral"] = True
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
    max_splits: int = 1,
) -> Dict[str, Any]:
    from src.config import load_config
    from src.data.datasets import build_registry
    from src.memory import cleanup_memory
    from src.train.experiment import run_single_file, set_seed

    # CRITICAL: pass hardware into load_config so cfg["hardware"] is a dict
    # (never assign the string "hardware_16gb.yaml" — that file does not exist).
    base = load_config(
        str(PROJECT_ROOT / "configs" / "adadae_per.yaml"),
        hardware=hardware,
    )
    if not isinstance(base.get("hardware"), dict):
        raise TypeError(
            f"config['hardware'] must be a dict after load_config, got {type(base.get('hardware'))}"
        )

    cands = candidates or list(STRIP_FIRST_CANDIDATES)
    adbench = Path(base["paths"]["adbench_root"])
    registry = build_registry(adbench)
    by_name = {e.name: e for e in registry}
    results: Dict[str, Any] = {
        "mode": "gpu_val_loss_select",
        "selection_metric": "val_loss",
        "winners": {},
        "jobs": [],
        "failures": [],
        "candidates": cands,
        "datasets": datasets,
        "hardware_profile": (base.get("hardware") or {}).get("hardware_profile"),
        "n_ok": 0,
        "n_fail": 0,
    }

    for ds in datasets:
        if ds not in by_name:
            print(f"SKIP missing dataset {ds}")
            results["failures"].append({"dataset": ds, "error": "missing_from_registry"})
            continue
        entry = by_name[ds]
        cat = entry.category  # registry truth
        rels = list(entry.relative_paths)[: max(1, max_splits)]
        if not rels:
            print(f"SKIP {ds}: no relative_paths")
            results["failures"].append({"dataset": ds, "error": "no_relative_paths"})
            continue

        ds_votes: Dict[str, List[float]] = {c: [] for c in cands}
        for seed in seeds:
            best_name = None
            best_val = float("inf")
            for cand in cands:
                cfg = _build_candidate_cfg(base, cand)
                set_seed(seed)
                split_vals: List[float] = []
                split_prs: List[float] = []
                last_exc: Optional[str] = None
                for rel in rels:
                    try:
                        out = run_single_file(
                            npz_path=adbench / rel,
                            setting="semi-supervised",
                            seed=seed,
                            config=cfg,
                            dataset_name=entry.name,
                            split_name=rel,
                            category=cat,
                        )
                    except Exception as exc:  # noqa: BLE001
                        last_exc = f"{type(exc).__name__}: {exc}"
                        print(f"FAIL {ds} {cand} seed={seed} split={rel}: {last_exc}")
                        results["failures"].append(
                            {
                                "dataset": ds,
                                "candidate": cand,
                                "seed": seed,
                                "split": rel,
                                "error": last_exc,
                            }
                        )
                        results["n_fail"] += 1
                        cleanup_memory()
                        continue
                    raw = out.get("best_val_metric")
                    if raw is None:
                        last_exc = "best_val_metric is None"
                        print(f"FAIL {ds} {cand} seed={seed}: {last_exc}")
                        results["n_fail"] += 1
                        cleanup_memory()
                        continue
                    val = float(raw)
                    split_vals.append(val)
                    pr = (out.get("metrics") or {}).get("PR-AUC")
                    if pr is not None:
                        split_prs.append(float(pr))
                    results["n_ok"] += 1
                    cleanup_memory()

                if not split_vals:
                    continue
                val = sum(split_vals) / len(split_vals)
                ds_votes[cand].append(val)
                results["jobs"].append(
                    {
                        "dataset": ds,
                        "seed": seed,
                        "candidate": cand,
                        "val_loss": val,
                        "PR": (sum(split_prs) / len(split_prs)) if split_prs else None,
                        "n_splits": len(split_vals),
                    }
                )
                if val < best_val:
                    best_val = val
                    best_name = cand
            if best_name:
                results.setdefault("_seed_winners", {}).setdefault(ds, []).append(best_name)

        means = {c: (sum(v) / len(v) if v else float("inf")) for c, v in ds_votes.items()}
        finite = {c: m for c, m in means.items() if m != float("inf")}
        if not finite:
            print(f"ERROR {ds}: all candidates failed (no finite val_loss)")
            results["winners"][ds] = {
                "policy": None,
                "mean_val_loss": means,
                "strip_mce_gate": True,
                "category": cat,
                "error": "all_candidates_failed",
            }
            continue
        winner = min(finite, key=finite.get)
        results["winners"][ds] = {
            "policy": winner,
            "mean_val_loss": means,
            "strip_mce_gate": True,
            "category": cat,
        }
    return results


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--preset",
        choices=["ship", "hard-12", "bleed-cv", "bleed-classical", "legacy"],
        default=None,
        help="Dataset preset (default ship when --datasets omitted and not dry-run)",
    )
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=[111, 222, 333])
    p.add_argument("--hardware", default=None)
    p.add_argument(
        "--max-splits",
        type=int,
        default=1,
        help="Max NPZ splits per CV/NLP family (default 1 for speed; protocol uses all)",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--apply-evidence-freeze", action="store_true")
    p.add_argument(
        "--candidates",
        nargs="*",
        default=None,
        help="Override strip-first candidate list",
    )
    args = p.parse_args()

    if args.dry_run:
        report = dry_run_resolve()
        out = PROJECT_ROOT / "results/adadae_per/thesis/loop6_train_only_select.json"
        rc = 0
    elif args.apply_evidence_freeze:
        report = evidence_freeze_report()
        out = PROJECT_ROOT / "results/adadae_per/thesis/phase1_hard_freeze.json"
        rc = 0
    else:
        if args.datasets:
            datasets = list(args.datasets)
        elif args.preset:
            datasets = _preset_datasets(args.preset)
        else:
            datasets = _preset_datasets("ship")
        report = run_selection(
            datasets,
            args.seeds,
            args.hardware,
            args.candidates,
            max_splits=args.max_splits,
        )
        out = PROJECT_ROOT / "results/adadae_per/thesis/phase1_hard_freeze.json"
        n_winners = sum(
            1
            for w in (report.get("winners") or {}).values()
            if isinstance(w, dict) and w.get("policy")
        )
        n_fail_ds = sum(
            1
            for w in (report.get("winners") or {}).values()
            if isinstance(w, dict) and w.get("error")
        )
        report["n_winners"] = n_winners
        report["n_failed_datasets"] = n_fail_ds
        if n_winners == 0:
            report["status"] = "FAILED_all_infinity"
            rc = 2
        elif n_fail_ds:
            report["status"] = "partial"
            rc = 1
        else:
            report["status"] = "ok"
            rc = 0

    out.parent.mkdir(parents=True, exist_ok=True)
    # JSON cannot encode Inf; replace for readability
    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, float) and (obj == float("inf") or obj != obj):
            return None
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    out.write_text(json.dumps(_sanitize(report), indent=2), encoding="utf-8")
    printable = {k: report[k] for k in report if k != "jobs"}
    print(json.dumps(_sanitize(printable), indent=2)[:4000])
    print(f"Wrote {out} status={report.get('status', 'ok')} n_ok={report.get('n_ok')} n_fail={report.get('n_fail')}")
    if rc != 0:
        print(
            "ERROR: selection produced no usable winners (check FAIL lines / hardware / ADBench paths).",
            file=sys.stderr,
        )
        print(
            "Delete bogus Infinity freeze before apply_hard_tail_freeze; "
            "fix then re-run: python scripts/train_only_recipe_select.py --preset ship --hardware 16gb",
            file=sys.stderr,
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
