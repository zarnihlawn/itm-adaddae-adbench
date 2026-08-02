#!/usr/bin/env python3
"""Full ADBench protocol: 57 datasets × 2 settings × 5 seeds (resumable)."""
from __future__ import annotations

import argparse
import json
import sys
import time
import fcntl
from contextlib import contextmanager
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
    if not path.exists():
        return {"completed": {}, "failed": {}}
    with open(path, "r", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        try:
            return json.load(f)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


@contextmanager
def _completed_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            raw = f.read()
            state = json.loads(raw) if raw.strip() else {"completed": {}, "failed": {}}
            yield state
            f.seek(0)
            f.truncate()
            json.dump(state, f, indent=2)
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def save_completed(path: Path, state: dict) -> None:
    with _completed_lock(path) as locked:
        locked.clear()
        locked.update(state)


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
            category=spec.category,
        )
        split_rows.append(row)
        cleanup_memory()

    agg = mean_std_metrics([r["metrics"] for r in split_rows])
    vram_vals = [
        float(r["vram_peak_mb"])
        for r in split_rows
        if r.get("vram_peak_mb") is not None
    ]
    if not vram_vals:
        vram_vals = [
            float(r["vram_mb"])
            for r in split_rows
            if r.get("vram_mb") is not None
        ]
    summary = {
        "dataset": spec.name,
        "setting": setting,
        "seed": seed,
        "n_splits": len(split_rows),
        "metrics_mean": {k: v["mean"] for k, v in agg.items()},
        "noise": split_rows[0].get("noise") if split_rows else {},
        "resolved_policy": split_rows[0].get("resolved_policy") if split_rows else None,
        "vram_peak_mb": max(vram_vals) if vram_vals else None,
        "vram_mb": max(vram_vals) if vram_vals else None,
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
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Optional seed subset (default: all seeds from config)",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Run shard K of N parallel workers (0-based). Jobs assigned by global index %% N.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total parallel workers when sharding (e.g. 2 for dual-GPU).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, hardware=args.hardware)
    seeds = list(args.seeds if args.seeds is not None else cfg.get("seeds", [111, 222, 333, 444, 555]))
    run_id = cfg["paths"].get("run_id", "adadae_full")
    results_dir = Path(cfg["paths"]["results_dir"])
    adbench = Path(cfg["paths"]["adbench_root"])

    completed_path = results_dir / "metrics" / "completed.json"

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

    if args.num_shards > 1:
        jobs = [job for i, job in enumerate(jobs) if i % args.num_shards == args.shard_index]

    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]

    total = len(jobs)
    shard_note = f" shard={args.shard_index}/{args.num_shards}" if args.num_shards > 1 else ""
    logger.info(f"Protocol start: {total} jobs{shard_note}", n_datasets=len(registry), seeds=seeds)
    print(f"Total jobs: {total}{shard_note} | Resume file: {completed_path}")

    bar = job_progress(total, desc="protocol")
    t0 = time.perf_counter()
    done = 0
    vram_watermark = 0.0
    for spec, setting, seed in jobs:
        key = job_key(spec.name, setting, seed)
        state = load_completed(completed_path)
        if key in state["completed"]:
            bar.update(1)
            done += 1
            continue
        try:
            logger.info(f"START {key}")
            summary = run_dataset_job(spec, setting, seed, cfg, logger, adbench)
            with _completed_lock(completed_path) as state:
                state.setdefault("completed", {})[key] = summary
                state.get("failed", {}).pop(key, None)
            # also write per-job json
            out = results_dir / "metrics" / f"{key}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            m = summary["metrics_mean"]
            done += 1
            elapsed = time.perf_counter() - t0
            rate = elapsed / max(done, 1)
            eta_s = rate * max(0, total - done)
            vram = summary.get("vram_peak_mb") or summary.get("vram_mb") or 0.0
            try:
                vram_watermark = max(vram_watermark, float(vram or 0.0))
            except (TypeError, ValueError):
                pass
            bar.set_postfix(
                job=key[:40],
                pr=f"{m.get('PR-AUC', float('nan')):.3f}",
                roc=f"{m.get('ROC-AUC', float('nan')):.3f}",
                eta_min=f"{eta_s / 60.0:.1f}",
                vram_mb=f"{vram_watermark:.0f}",
            )
        except Exception as e:
            logger.log("job_failed", key=key, error=str(e))
            with _completed_lock(completed_path) as state:
                state.setdefault("failed", {})[key] = str(e)
            print(f"FAILED {key}: {e}")
            done += 1
        bar.update(1)
        cleanup_memory()

    bar.close()
    state = load_completed(completed_path)
    logger.info(
        "Protocol finished",
        n_completed=len(state["completed"]),
        n_failed=len(state.get("failed", {})),
        vram_watermark_mb=vram_watermark,
    )
    logger.close()
    print(
        f"Done. Completed={len(state['completed'])} Failed={len(state.get('failed', {}))} "
        f"VRAM watermark≈{vram_watermark:.0f} MB"
    )


if __name__ == "__main__":
    main()
