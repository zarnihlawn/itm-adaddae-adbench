#!/usr/bin/env python3
"""Run AdaDDAE on a single ADBench dataset (averages multi-split families)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.datasets import get_spec
from src.eval.metrics import mean_std_metrics
from src.runlog.logger import RunLogger
from src.train.experiment import run_single_file


def main():
    parser = argparse.ArgumentParser(description="Run AdaDDAE on one dataset")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--hardware",
        type=str,
        default=None,
        help="Override hardware profile: 8gb|12gb|16gb|rtx5070ti|hardware_rtx5070ti.yaml",
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. cardio or CIFAR10")
    parser.add_argument("--setting", type=str, default=None, choices=["unsupervised", "semi-supervised"])
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, hardware=args.hardware)
    setting = args.setting or cfg["train"]["setting"]
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 111))
    run_id = cfg["paths"].get("run_id", "adadae")
    results_dir = Path(cfg["paths"]["results_dir"])
    log_path = results_dir / "logs" / f"{run_id}_{args.dataset}_{setting}_{seed}.jsonl"
    logger = RunLogger(log_path, run_id=run_id)

    adbench = Path(cfg["paths"]["adbench_root"])
    spec = get_spec(adbench, args.dataset)

    split_rows = []
    for rel in spec.relative_paths:
        npz_path = adbench / rel
        logger.info(f"Running {spec.name} :: {rel}", setting=setting, seed=seed)
        row = run_single_file(
            npz_path=npz_path,
            setting=setting,
            seed=seed,
            config=cfg,
            logger=logger,
            dataset_name=spec.name,
            split_name=rel,
        )
        split_rows.append(row)

    # Average metrics across splits (for multi-file families)
    agg = mean_std_metrics([r["metrics"] for r in split_rows])
    summary = {
        "dataset": spec.name,
        "setting": setting,
        "seed": seed,
        "n_splits": len(split_rows),
        "metrics_mean": {k: v["mean"] for k, v in agg.items()},
        "metrics_std_across_splits": {k: v["std"] for k, v in agg.items()},
        "splits": split_rows,
    }

    out_dir = results_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{spec.name}__{setting}__{seed}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary ===")
    for k, v in summary["metrics_mean"].items():
        print(f"{k}: {v:.4f}")
    print(f"Saved: {out_path}")
    logger.close()


if __name__ == "__main__":
    main()
