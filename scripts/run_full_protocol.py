#!/usr/bin/env python3
"""Full ADBench protocol: 57 datasets × 2 settings × 5 seeds (resumable)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.datasets import build_registry
from src.eval.metrics import mean_std_metrics
from src.memory import cleanup_memory
from src.runlog.logger import RunLogger, job_progress
from src.train.experiment import run_single_file


def job_key(dataset: str, setting: str, seed: int) -> str:
    return f"{dataset}__{setting}__{seed}"


def load_completed(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": {}, "failed": {}}


def save_completed(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def run_dataset_job(spec, setting, seed, cfg, logger, adbench):
    split_rows = []
    for rel in spec.relative_paths:
        row = run_single_file(
            npz_path=adbench / rel,
            setting=setting,
            seed=seed,
            config=cfg,
            logger=logger,
            dataset_name=spec.name,
            split_name=rel,
        )
        split_rows.append(row)
        cleanup_memory()

    agg = mean_std_metrics([r["metrics"] for r in split_rows])
    summary = {
        "dataset": spec.name,
        "setting": setting,
        "seed": seed,
        "n_splits": len(split_rows),
        "metrics_mean": {k: v["mean"] for k, v in agg.items()},
        "noise": split_rows[0].get("noise") if split_rows else {},
        "time": datetime.now(timezone.utc).isoformat(),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Full AdaDDAE ADBench protocol")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--hardware",
        type=str,
        default=None,
        help="Override hardware profile: 8gb|12gb|16gb|rtx5070ti|hardware_rtx5070ti.yaml",
    )
    parser.add_argument("--settings", nargs="+", default=["unsupervised", "semi-supervised"])
    parser.add_argument("--max-jobs", type=int, default=None, help="Optional cap for smoke tests")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional subset of dataset names")
    args = parser.parse_args()

    cfg = load_config(args.config, hardware=args.hardware)
    seeds = list(cfg.get("seeds", [111, 222, 333, 444, 555]))
    run_id = cfg["paths"].get("run_id", "adadae_full")
    results_dir = Path(cfg["paths"]["results_dir"])
    adbench = Path(cfg["paths"]["adbench_root"])

    completed_path = results_dir / "metrics" / "completed.json"
    state = load_completed(completed_path)

    log_path = results_dir / "logs" / f"{run_id}.jsonl"
    logger = RunLogger(log_path, run_id=run_id)

    registry = build_registry(adbench)
    if args.datasets:
        wanted = {d.lower() for d in args.datasets}
        registry = [s for s in registry if s.name.lower() in wanted]

    jobs = []
    for spec in registry:
        for setting in args.settings:
            for seed in seeds:
                jobs.append((spec, setting, seed))

    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]

    total = len(jobs)
    logger.info(f"Protocol start: {total} jobs", n_datasets=len(registry), seeds=seeds)
    print(f"Total jobs: {total} | Resume file: {completed_path}")

    bar = job_progress(total, desc="protocol")
    for spec, setting, seed in jobs:
        key = job_key(spec.name, setting, seed)
        if key in state["completed"]:
            bar.update(1)
            continue
        try:
            logger.info(f"START {key}")
            summary = run_dataset_job(spec, setting, seed, cfg, logger, adbench)
            state["completed"][key] = summary
            # also write per-job json
            out = results_dir / "metrics" / f"{key}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            save_completed(completed_path, state)
            m = summary["metrics_mean"]
            bar.set_postfix(
                job=key[:40],
                pr=f"{m.get('PR-AUC', float('nan')):.3f}",
                roc=f"{m.get('ROC-AUC', float('nan')):.3f}",
            )
        except Exception as e:
            logger.log("job_failed", key=key, error=str(e))
            state.setdefault("failed", {})[key] = str(e)
            save_completed(completed_path, state)
            print(f"FAILED {key}: {e}")
        bar.update(1)
        cleanup_memory()

    bar.close()
    logger.info("Protocol finished", n_completed=len(state["completed"]), n_failed=len(state.get("failed", {})))
    logger.close()
    print(f"Done. Completed={len(state['completed'])} Failed={len(state.get('failed', {}))}")


if __name__ == "__main__":
    main()
