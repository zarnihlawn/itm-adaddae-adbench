#!/usr/bin/env python3
"""Benchmark GPU throughput and peak VRAM on representative ADBench jobs."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.config import load_config
from src.data.datasets import get_spec
from src.train.experiment import run_single_file

BENCH_DATASETS = [
    ("cardio", "semi-supervised"),
    ("cardio", "unsupervised"),
    ("ALOI", "unsupervised"),
    ("thyroid", "semi-supervised"),
    ("breastw", "semi-supervised"),
    ("pendigits", "semi-supervised"),
    ("vowels", "unsupervised"),
    ("musk", "semi-supervised"),
    ("satellite", "semi-supervised"),
    ("annthyroid", "semi-supervised"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default_gpu.yaml")
    parser.add_argument(
        "--hardware",
        type=str,
        default=None,
        help="Override hardware profile: 8gb|12gb|16gb|rtx5070ti",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=111)
    args = parser.parse_args()

    cfg = load_config(args.config, hardware=args.hardware)
    cfg["train"]["epochs"] = args.epochs
    cfg["train"]["eval_every"] = max(5, args.epochs // 3)
    adbench = Path(cfg["paths"]["adbench_root"])
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    times = []
    vram_peaks = []
    rows = []

    for ds_name, setting in BENCH_DATASETS:
        spec = get_spec(adbench, ds_name)
        rel = spec.relative_paths[0]
        t0 = time.time()
        row = run_single_file(
            npz_path=adbench / rel,
            setting=setting,
            seed=args.seed,
            config=cfg,
            dataset_name=ds_name,
            split_name=rel,
        )
        elapsed = time.time() - t0
        times.append(elapsed)
        vram = row.get("vram_mb") or 0.0
        if torch.cuda.is_available():
            vram = max(vram, torch.cuda.max_memory_allocated() / 1024**2)
        vram_peaks.append(vram)
        rows.append({
            "dataset": ds_name,
            "setting": setting,
            "seconds": elapsed,
            "vram_mb": vram,
            "pr_auc": row["metrics"]["PR-AUC"],
            "roc_auc": row["metrics"]["ROC-AUC"],
        })
        print(f"{ds_name}/{setting}: {elapsed:.1f}s PR-AUC={row['metrics']['PR-AUC']:.4f} VRAM={vram:.0f}MB")

    jobs_per_hour = 3600.0 / (sum(times) / len(times)) if times else 0.0
    summary = {
        "n_jobs": len(times),
        "total_seconds": sum(times),
        "mean_seconds_per_job": sum(times) / max(len(times), 1),
        "jobs_per_hour_estimate": jobs_per_hour,
        "peak_vram_mb": max(vram_peaks) if vram_peaks else 0.0,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "jobs": rows,
    }

    out = results_dir / "thesis" / "benchmark_gpu.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nMean {summary['mean_seconds_per_job']:.1f}s/job (~{jobs_per_hour:.1f} jobs/hr)")
    print(f"Peak VRAM {summary['peak_vram_mb']:.0f} MB")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
