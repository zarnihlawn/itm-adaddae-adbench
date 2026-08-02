#!/usr/bin/env python3
"""Profile ftp / train / score / geode / fusion timing breakdown → JSON + CSV."""
from __future__ import annotations

import argparse
import csv
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
    parser.add_argument("--hardware", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, hardware=args.hardware)
    cfg["train"]["epochs"] = args.epochs
    cfg["train"]["eval_every"] = args.epochs
    cfg.setdefault("adadae", {})["profile_breakdown"] = True
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
        prof = row.get("profile") or {}
        entry = {
            "dataset": ds_name,
            "setting": setting,
            "ftp_sec": row.get("ftp_sec", 0),
            "train_sec": row.get("train_sec", 0),
            "score_sec": row.get("score_sec", 0),
            "geode_sec": prof.get("geode", 0),
            "fusion_sec": prof.get("fusion", 0),
            "orbis_sec": prof.get("orbis", 0),
            "plexus_sec": prof.get("plexus", 0),
            "total_sec": row.get("total_sec", 0),
            "vram_peak_mb": row.get("vram_peak_mb"),
            "PR-AUC": row["metrics"]["PR-AUC"],
            "score_timesteps": row.get("score_timesteps"),
        }
        rows.append(entry)
        print(
            f"{ds_name}/{setting}: ftp={entry['ftp_sec']:.2f}s "
            f"train={entry['train_sec']:.2f}s score={entry['score_sec']:.2f}s "
            f"geode={entry['geode_sec']:.2f}s fusion={entry['fusion_sec']:.2f}s "
            f"PR={entry['PR-AUC']:.4f}"
        )

    out_dir = results_dir / "thesis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "profile_breakdown.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"jobs": rows}, f, indent=2)
    out_csv = out_dir / "profile_breakdown.csv"
    if rows:
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
