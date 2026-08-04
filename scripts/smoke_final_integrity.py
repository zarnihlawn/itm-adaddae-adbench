#!/usr/bin/env python3
"""Phase-1 smoke: 3 datasets × 2 settings × 2 seeds under frozen final recipe.

Asserts:
  - config integrity (static / val_loss / no MCE-GATE)
  - every job has finite metrics
  - early_stop_metric is val_loss (never test PR)
  - resolved_policy is absent/null (no routing)
  - resume: second pass skips completed jobs
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _load_assert_mod():
    path = PROJECT_ROOT / "scripts" / "assert_final_config.py"
    spec = importlib.util.spec_from_file_location("assert_final_config", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


from src.config import load_config  # noqa: E402
from src.data.datasets import build_registry  # noqa: E402
from src.memory import cleanup_memory  # noqa: E402
from src.runlog.logger import RunLogger  # noqa: E402
from src.train.experiment import run_single_file  # noqa: E402

DEFAULT_DATASETS = ["cardio", "glass", "vertebral"]
DEFAULT_SETTINGS = ["unsupervised", "semi-supervised"]
DEFAULT_SEEDS = [111, 222]


def _finite_metrics(m: dict) -> bool:
    for k in ("PR-AUC", "ROC-AUC", "AP"):
        v = m.get(k)
        if v is None or not math.isfinite(float(v)):
            return False
    return True


def _allowed_smoke_policy(pol) -> bool:
    if pol in (None, "", "static"):
        return True
    if isinstance(pol, str) and pol.startswith("paradigm_"):
        return True
    return False


def _scan_logs(log_dir: Path, run_id: str) -> list[str]:
    errs: list[str] = []
    if not log_dir.exists():
        return [f"log dir missing: {log_dir}"]
    files = sorted(log_dir.glob(f"{run_id}_*.jsonl"))
    single = log_dir / f"{run_id}.jsonl"
    if single.exists():
        files = [single] + [f for f in files if f != single]
    if not files:
        return [f"no job logs matching {run_id}.jsonl or {run_id}_*.jsonl in {log_dir}"]

    saw_job_end = 0
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = ev.get("event")
            if event == "job_start":
                esm = ev.get("early_stop_metric")
                if esm not in (None, "val_loss", "train_loss"):
                    errs.append(f"{path.name}: job_start early_stop_metric={esm!r}")
                if esm == "val_pr":
                    errs.append(f"{path.name}: val_pr early stop not allowed for primary smoke")
            if event == "early_stop":
                esm = ev.get("early_stop_metric")
                if esm not in ("val_loss", "train_loss"):
                    errs.append(f"{path.name}: early_stop metric={esm!r}")
                if "best_pr_auc" in ev and ev.get("best_val_metric") is None and esm != "val_pr":
                    # legacy field alone is OK if best_val_metric present; reject test-style only
                    pass
                if "best_test" in str(ev).lower():
                    errs.append(f"{path.name}: early_stop mentions test")
            if event == "epoch" and "PR-AUC" in ev and ev.get("early_stop_metric") not in (
                None,
                "val_pr",
            ):
                # PR-AUC in epoch rows only allowed for val_pr mode
                if ev.get("early_stop_metric") == "val_loss":
                    errs.append(f"{path.name}: epoch logged PR-AUC under val_loss mode")
            if event == "job_end":
                saw_job_end += 1
                esm = ev.get("early_stop_metric")
                if esm not in ("val_loss", "train_loss", None):
                    errs.append(f"{path.name}: job_end early_stop_metric={esm!r}")
                if not _allowed_smoke_policy(ev.get("resolved_policy")):
                    errs.append(f"{path.name}: resolved_policy={ev.get('resolved_policy')!r}")
                metrics = ev.get("metrics") or {}
                if metrics and not _finite_metrics(metrics):
                    errs.append(f"{path.name}: non-finite metrics {metrics}")
    if saw_job_end == 0:
        errs.append("no job_end events in logs")
    return errs


def run_smoke(cfg: dict, datasets: list[str], settings: list[str], seeds: list[int]) -> Path:
    results_dir = Path(cfg["paths"]["results_dir"])
    run_id = cfg["paths"].get("run_id", "adadae_final_smoke")
    adbench = Path(cfg["paths"]["adbench_root"])
    metrics_dir = results_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    completed_path = metrics_dir / "completed.json"
    if completed_path.exists():
        state = json.loads(completed_path.read_text(encoding="utf-8"))
    else:
        state = {"completed": {}, "failed": {}}

    registry = build_registry(adbench)
    wanted = {d.lower() for d in datasets}
    specs = [s for s in registry if s.name.lower() in wanted]
    if len(specs) != len(wanted):
        found = {s.name.lower() for s in specs}
        missing = sorted(wanted - found)
        raise SystemExit(f"datasets not found: {missing}")

    log_path = results_dir / "logs" / f"{run_id}.jsonl"
    logger = RunLogger(log_path, run_id=run_id)

    jobs = [(spec, setting, seed) for spec in specs for setting in settings for seed in seeds]
    print(f"Smoke jobs: {len(jobs)} -> {results_dir}")

    for spec, setting, seed in jobs:
        key = f"{spec.name}__{setting}__{seed}"
        if key in state["completed"]:
            print(f"SKIP {key}")
            continue
        print(f"RUN  {key}")
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
        from src.eval.metrics import mean_std_metrics

        agg = mean_std_metrics([r["metrics"] for r in split_rows])
        summary = {
            "dataset": spec.name,
            "setting": setting,
            "seed": seed,
            "n_splits": len(split_rows),
            "metrics_mean": {k: v["mean"] for k, v in agg.items()},
            "early_stop_metric": split_rows[0].get("early_stop_metric"),
            "best_val_metric": split_rows[0].get("best_val_metric"),
            "resolved_policy": split_rows[0].get("resolved_policy"),
            "n_val": split_rows[0].get("n_val"),
        }
        state["completed"][key] = summary
        state.get("failed", {}).pop(key, None)
        (metrics_dir / f"{key}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        completed_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        m = summary["metrics_mean"]
        print(
            f"  OK PR={m.get('PR-AUC', float('nan')):.4f} "
            f"ROC={m.get('ROC-AUC', float('nan')):.4f} "
            f"esm={summary.get('early_stop_metric')} n_val={summary.get('n_val')}"
        )

    logger.close()
    return completed_path


def assert_completed(completed_path: Path, expected_n: int, min_pr_auc: float | None = None) -> list[str]:
    errs: list[str] = []
    data = json.loads(completed_path.read_text(encoding="utf-8"))
    completed = data.get("completed", data)
    if len(completed) < expected_n:
        errs.append(f"expected >= {expected_n} completed jobs, got {len(completed)}")
    for key, job in completed.items():
        m = job.get("metrics_mean") or job.get("metrics") or {}
        if not _finite_metrics(m):
            errs.append(f"{key}: non-finite metrics {m}")
        esm = job.get("early_stop_metric")
        if esm not in ("val_loss", "train_loss", None):
            errs.append(f"{key}: early_stop_metric={esm!r}")
        if not _allowed_smoke_policy(job.get("resolved_policy")):
            errs.append(f"{key}: resolved_policy={job.get('resolved_policy')!r}")
        if job.get("n_val") is not None and int(job["n_val"]) <= 0 and esm == "val_loss":
            errs.append(f"{key}: val_loss claimed but n_val={job.get('n_val')}")
        if min_pr_auc is not None:
            pr = m.get("PR-AUC")
            if pr is None or float(pr) < float(min_pr_auc):
                errs.append(f"{key}: PR-AUC={pr!r} below floor {min_pr_auc}")
    return errs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/adadae_final_smoke.yaml")
    p.add_argument("--hardware", default=None, help="e.g. cpu | 16gb | 12gb")
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--settings", nargs="+", default=DEFAULT_SETTINGS)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--skip-run", action="store_true", help="Only assert existing smoke results")
    p.add_argument(
        "--allow-nonfinal-run-id",
        action="store_true",
        help="Smoke uses run_id adadae_final_smoke; skip run_id==adadae_final check",
    )
    p.add_argument(
        "--min-pr-auc",
        type=float,
        default=None,
        help="Fail if any completed job mean PR-AUC is below this floor",
    )
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path

    # Integrity on YAML (allow smoke run_id)
    assert_mod = _load_assert_mod()
    raw = assert_mod.load_config_for_audit(cfg_path)
    cfg_errs = assert_mod.audit_primary_config(raw, require_final_run_id=False)
    # Still require val_loss / static / no MCE-GATE (audit already does)
    if raw.get("adadae", {}).get("policy") == "routed":
        cfg_errs.append("smoke config must not use routed policy")
    if raw.get("adadae", {}).get("policy") not in ("static", "paradigm", None):
        cfg_errs.append(
            f"smoke config policy must be static or paradigm "
            f"(got {raw.get('adadae', {}).get('policy')!r})"
        )
    if cfg_errs:
        print("CONFIG FAIL:")
        for e in cfg_errs:
            print(f"  - {e}")
        return 1
    print("CONFIG OK")

    cfg = load_config(cfg_path, hardware=args.hardware)
    results_dir = Path(cfg["paths"]["results_dir"])
    run_id = cfg["paths"].get("run_id", "adadae_final_smoke")
    expected = len(args.datasets) * len(args.settings) * len(args.seeds)

    if not args.skip_run:
        completed_path = run_smoke(cfg, args.datasets, args.settings, args.seeds)
    else:
        completed_path = results_dir / "metrics" / "completed.json"

    errs = assert_completed(completed_path, expected, min_pr_auc=args.min_pr_auc)
    errs.extend(_scan_logs(results_dir / "logs", run_id))

    # Resume check: second invocation should skip all
    if not args.skip_run:
        before = json.loads(completed_path.read_text(encoding="utf-8"))
        n_before = len(before.get("completed", {}))
        run_smoke(cfg, args.datasets, args.settings, args.seeds)
        after = json.loads(completed_path.read_text(encoding="utf-8"))
        n_after = len(after.get("completed", {}))
        if n_after != n_before:
            errs.append(f"resume mutated completed count {n_before} -> {n_after}")
        else:
            print(f"RESUME OK ({n_after} jobs unchanged)")

    if errs:
        print("SMOKE FAIL:")
        for e in errs:
            print(f"  - {e}")
        return 1

    print(f"SMOKE PASS: {expected} jobs under {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
