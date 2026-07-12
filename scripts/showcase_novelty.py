#!/usr/bin/env python3
"""Novelty showcase: ablation waterfall on 5 canonical datasets x both settings."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ablations import ABLATIONS, LADDER_ORDER, deep_update
from src.config import load_config
from src.data.datasets import get_spec
from src.eval.metrics import mean_std_metrics
from src.memory import cleanup_memory
from src.train.experiment import run_single_file

SHOWCASE_DATASETS = ["cardio", "ALOI", "thyroid", "breastw", "pendigits"]
SHOWCASE_STEPS = [
    "ddae_repro",
    "adadae_fixed",
    "ftp",
    "lfdanc",
    "ssts",
    "rdt",
    "taps",
    "vus",
    "dte",
    "full_adadae",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ablation_ladder.yaml")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    base = load_config(args.config)
    base["train"]["epochs"] = args.epochs
    base["train"]["eval_every"] = max(5, args.epochs // 4)
    adbench = Path(base["paths"]["adbench_root"])
    results_dir = Path(base["paths"]["results_dir"])

    all_rows = []
    for setting in ["unsupervised", "semi-supervised"]:
        baseline_pr = None
        for step in SHOWCASE_STEPS:
            cfg = deep_update(base, ABLATIONS[step])
            ds_metrics = []
            for ds_name in SHOWCASE_DATASETS:
                spec = get_spec(adbench, ds_name)
                rel = spec.relative_paths[0]
                row = run_single_file(
                    npz_path=adbench / rel,
                    setting=setting,
                    seed=args.seed,
                    config=cfg,
                    dataset_name=ds_name,
                    split_name=rel,
                )
                ds_metrics.append(row["metrics"])
                cleanup_memory()
            agg = mean_std_metrics(ds_metrics)
            pr = agg["PR-AUC"]["mean"]
            roc = agg["ROC-AUC"]["mean"]
            if step == "ddae_repro":
                baseline_pr = pr
            delta = (pr - baseline_pr) * 100 if baseline_pr is not None else 0.0
            all_rows.append({
                "setting": setting,
                "step": step,
                "PR-AUC_pct": pr * 100,
                "ROC-AUC_pct": roc * 100,
                "delta_PR_pct": delta,
                "cumulative_delta_PR_pct": delta,
            })
            print(f"{setting}/{step}: PR={pr*100:.2f}% Δ={delta:+.2f}")

    df = pd.DataFrame(all_rows)
    out_dir = results_dir / "thesis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "novelty_showcase.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)
    df.to_csv(out_dir / "novelty_showcase.csv", index=False)
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
