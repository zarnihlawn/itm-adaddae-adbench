#!/usr/bin/env python3
"""Val-only hyperparameter search for AdaDDAE-5 (never uses test PR)."""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.datasets import build_registry
from src.memory import cleanup_memory
from src.train.experiment import run_single_file

GRID = {
    "dsm_plus_lambda": [0.3, 0.5, 0.7],
    "ib_latent_beta": [0.005, 0.01],
    "vmf_kappa": [0.5, 1.0],
    "scs_max_timesteps": [32, 64],
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/adadae5_final.yaml")
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
    base["paths"]["run_id"] = "adadae5_tune"
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
            best_val = mean_val
            best = trial

    out_dir = PROJECT_ROOT / "results" / "adadae5_tune"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tune_audit.json").write_text(json.dumps({"best": best, "audit": audit}, indent=2))
    print(f"best={best} mean_val_loss={best_val:.6f}")
    if args.write and best:
        import yaml

        cfg_path = PROJECT_ROOT / "configs" / "adadae5_final.yaml"
        with open(cfg_path) as f:
            doc = yaml.safe_load(f)
        doc.setdefault("adadae", {}).update(best)
        with open(cfg_path, "w") as f:
            yaml.safe_dump(doc, f, sort_keys=False)
        print(f"wrote {best} into {cfg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
