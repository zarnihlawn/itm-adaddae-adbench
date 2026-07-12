#!/usr/bin/env python3
"""Profile ftp / train / score timing breakdown on representative datasets."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.datasets import get_spec
from src.train.experiment import run_single_file

PROFILE_DATASETS = [
    ("cardio", "semi-supervised"),
    ("ALOI", "unsupervised"),
    ("thyroid", "semi-supervised"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["train"]["epochs"] = args.epochs
    cfg["train"]["eval_every"] = args.epochs
    adbench = Path(cfg["paths"]["adbench_root"])
    results_dir = Path(cfg["paths"]["results_dir"])

    rows = []
    for ds_name, setting in PROFILE_DATASETS:
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
        rows.append({
            "dataset": ds_name,
            "setting": setting,
            "ftp_sec": row.get("ftp_sec", 0),
            "train_sec": row.get("train_sec", 0),
            "score_sec": row.get("score_sec", 0),
            "total_sec": row.get("total_sec", 0),
            "PR-AUC": row["metrics"]["PR-AUC"],
            "score_timesteps": row.get("score_timesteps"),
        })
        print(
            f"{ds_name}/{setting}: ftp={row.get('ftp_sec',0):.2f}s "
            f"train={row.get('train_sec',0):.2f}s score={row.get('score_sec',0):.2f}s "
            f"PR={row['metrics']['PR-AUC']:.4f}"
        )

    out = results_dir / "thesis" / "profile_breakdown.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"jobs": rows}, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
