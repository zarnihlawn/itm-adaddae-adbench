#!/usr/bin/env python3
"""Loop 6: train-only multi-recipe arbitration for hard semi datasets.

For each dataset×seed, train candidate recipes and pick the winner by
**val_loss** (never test PR). Emit frozen specialist recommendations into
a JSON that can be merged into adadae_per_exceptions.yaml after review.

Usage (GPU):
  python scripts/train_only_recipe_select.py --datasets glass Wilt vertebral --seeds 111 222

Compare-only / dry-run (resolve policies only):
  python scripts/train_only_recipe_select.py --dry-run
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

HARD_SEMI_DEFAULT = [
    "glass",
    "CIFAR10",
    "WBC",
    "Wilt",
    "vertebral",
    "Hepatitis",
    "mammography",
    "fraud",
    "Waveform",
    "speech",
    "20newsgroups",
    "cover",
    "optdigits",
]

CANDIDATES = [
    "baseline_ddae",
    "semi_cvnlp_ftp",
    "semi_rdt_tail",
    "semi_smc_tail",
]


def _category_for(name: str) -> str:
    """Best-effort category without requiring ADBench on disk."""
    cv = {"CIFAR10", "SVHN", "MNIST-C", "FashionMNIST", "MVTec-AD", "celeba", "ALOI", "speech"}
    nlp = {"Agnews", "Amazon", "Imdb", "Yelp", "20newsgroups"}
    if name in cv:
        return "cv"
    if name in nlp:
        return "nlp"
    return "classical"


def dry_run_resolve() -> Dict[str, Any]:
    from src.config import load_yaml
    from src.policy_per import apply_per_config

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
                "use_nautilus": out["adadae"].get("use_nautilus"),
                "use_apex": out["adadae"].get("use_apex"),
                "fusion_mode": out["adadae"].get("fusion_mode"),
            }
        )
    return {"mode": "dry_run", "candidates": CANDIDATES, "current_per": rows}


def run_selection(datasets: List[str], seeds: List[int], hardware: str | None) -> Dict[str, Any]:
    """Train candidates; select by lowest val_loss (train-only)."""
    from src.config import load_config
    from src.data.datasets import build_registry
    from src.policy import policy_overrides, _deep_update
    from src.train.experiment import run_single_file, set_seed

    base = load_config(str(PROJECT_ROOT / "configs" / "adadae_per.yaml"))
    if hardware:
        base["hardware"] = f"hardware_{hardware}.yaml" if not hardware.endswith(".yaml") else hardware
    # Short but integrity-preserving for probe
    base.setdefault("train", {})
    # Keep min_epochs from PER but allow probe override via env if needed

    registry = build_registry(base["paths"]["adbench_root"])
    by_name = {e.name: e for e in registry}
    results: Dict[str, Any] = {"winners": {}, "jobs": []}

    for ds in datasets:
        if ds not in by_name:
            print(f"SKIP missing dataset {ds}")
            continue
        entry = by_name[ds]
        cat = entry.category
        ds_votes: Dict[str, List[float]] = {c: [] for c in CANDIDATES}
        for seed in seeds:
            best_name = None
            best_val = float("inf")
            for cand in CANDIDATES:
                cfg = copy.deepcopy(base)
                cfg["adadae"] = dict(cfg.get("adadae") or {})
                # Force static candidate (bypass PER) for fair recipe compare
                cfg["adadae"]["policy"] = "static"
                cfg["adadae"].pop("exceptions_file", None)
                cfg["adadae"].pop("upgrades_file", None)
                cfg = _deep_update(cfg, policy_overrides(cand))
                cfg["adadae"]["resolved_policy"] = f"select:{cand}"
                cfg["paths"] = dict(cfg["paths"])
                cfg["paths"]["results_dir"] = "results/adadae_per_select"
                cfg["paths"]["run_id"] = "adadae_per_select"
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
                # Selection uses val_loss only (ignore PR for arbitration)
                if val < best_val:
                    best_val = val
                    best_name = cand
            if best_name:
                results.setdefault("_seed_winners", {}).setdefault(ds, []).append(best_name)

        # Majority vote across seeds by mean val_loss
        means = {
            c: (sum(v) / len(v) if v else float("inf")) for c, v in ds_votes.items()
        }
        winner = min(means, key=means.get) if means else "baseline_ddae"
        results["winners"][ds] = {"policy": winner, "mean_val_loss": means}
    return results


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="*", default=HARD_SEMI_DEFAULT)
    p.add_argument("--seeds", nargs="*", type=int, default=[111, 222])
    p.add_argument("--hardware", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.dry_run:
        report = dry_run_resolve()
    else:
        report = run_selection(args.datasets, args.seeds, args.hardware)

    out = PROJECT_ROOT / "results/adadae_per/thesis/loop6_train_only_select.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "jobs"}, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
