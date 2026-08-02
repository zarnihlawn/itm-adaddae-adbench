#!/usr/bin/env python3
"""Val-only hyperparameter search for AdaDDAE-4 (never uses test PR)."""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.datasets import build_registry
from src.memory import cleanup_memory
from src.train.experiment import run_single_file

GRID = {
    "chronos_hidden": [16, 32],
    "geode_neighbors": [16, 32],
    "aether_loss_weight": [0.05, 0.1],
    "nexus_loss_weight": [0.02, 0.05],
    "scs_max_timesteps": [32, 64],
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/adadae4_final.yaml")
    p.add_argument("--hardware", default=None)
    p.add_argument("--datasets", nargs="+", default=["cardio", "thyroid", "letter"])
    p.add_argument("--seeds", nargs="+", type=int, default=[111])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--write", action="store_true")
    p.add_argument("--max-trials", type=int, default=8)
    args = p.parse_args()

    base = load_config(args.config, hardware=args.hardware)
    base["train"]["epochs"] = args.epochs
    base["train"]["eval_every"] = max(2, args.epochs // 4)
    base["paths"]["run_id"] = "adadae4_tune"
    adbench = Path(base["paths"]["adbench_root"])
    registry = [s for s in build_registry(adbench) if s.name.lower() in {d.lower() for d in args.datasets}]

    keys = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    step = max(1, len(combos) // max(1, args.max_trials))
    combos = combos[::step][: args.max_trials]

    audit, best, best_val = [], None, float("inf")
    for combo in combos:
        cfg = copy.deepcopy(base)
        trial = dict(zip(keys, combo))
        for k, v in trial.items():
            cfg.setdefault("adadae", {})[k] = v
        val_losses = []
        for spec in registry:
            for seed in args.seeds:
                rel = spec.relative_paths[0]
                row = run_single_file(
                    npz_path=adbench / rel,
                    setting="semi-supervised",
                    seed=seed,
                    config=cfg,
                    dataset_name=spec.name,
                    split_name=rel,
                    category=spec.category,
                )
                bv = row.get("best_val_metric")
                if bv is not None and bv == bv:
                    val_losses.append(float(bv))
                cleanup_memory()
        mean_val = float(sum(val_losses) / max(1, len(val_losses))) if val_losses else float("inf")
        audit.append({"trial": trial, "mean_val_loss": mean_val, "n": len(val_losses)})
        print(f"trial {trial} mean_val_loss={mean_val:.6f}")
        if mean_val < best_val:
            best_val, best = mean_val, trial

    out_dir = Path(base["paths"]["results_dir"]) / "thesis"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "adadae4_valonly_tune.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"best": best, "best_val_loss": best_val, "trials": audit}, f, indent=2)
    print(f"Best: {best} val_loss={best_val:.6f}\nWrote {path}")
    if args.write and best:
        yp = PROJECT_ROOT / "configs" / "adadae4_final.yaml"
        with open(yp, encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        for k, v in best.items():
            doc.setdefault("adadae", {})[k] = v
        with open(yp, "w", encoding="utf-8") as f:
            yaml.safe_dump(doc, f, sort_keys=False)
        print(f"Updated {yp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
