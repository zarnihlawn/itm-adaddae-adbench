#!/usr/bin/env python3
"""LODO holdout eval of a frozen recipe (no design leakage disclosure helper)."""
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
from src.memory import cleanup_memory
from src.runlog.logger import RunLogger
from src.train.experiment import run_single_file

DEFAULT_HOLDOUT = ["speech", "Agnews", "Wilt", "celeba", "cardio"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/adadae_final.yaml")
    p.add_argument("--hardware", default="cpu")
    p.add_argument("--holdout", nargs="+", default=DEFAULT_HOLDOUT)
    p.add_argument("--seeds", nargs="+", type=int, default=[111, 222])
    p.add_argument("--settings", nargs="+", default=["unsupervised", "semi-supervised"])
    p.add_argument("--out", default="results/adadae_final/thesis/lodo_holdout.json")
    args = p.parse_args()

    cfg = load_config(args.config, hardware=args.hardware)
    adbench = Path(cfg["paths"]["adbench_root"])
    results = []
    for ds in args.holdout:
        try:
            spec = get_spec(adbench, ds)
        except Exception as e:
            results.append({"dataset": ds, "error": str(e)})
            continue
        for setting in args.settings:
            for seed in args.seeds:
                logger = RunLogger(
                    Path(cfg["paths"]["results_dir"]) / "logs" / f"lodo_{ds}_{setting}_{seed}.jsonl",
                    run_id=cfg["paths"].get("run_id", "lodo"),
                )
                rows = []
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
                    rows.append(row)
                    cleanup_memory()
                agg = mean_std_metrics([r["metrics"] for r in rows])
                results.append(
                    {
                        "dataset": spec.name,
                        "setting": setting,
                        "seed": seed,
                        "metrics_mean": {k: v["mean"] for k, v in agg.items()},
                        "early_stop_metric": rows[0].get("early_stop_metric"),
                    }
                )
                logger.close()
                print(f"LODO {spec.name} {setting} {seed} PR={results[-1]['metrics_mean'].get('PR-AUC', float('nan')):.4f}")

    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"holdout": args.holdout, "results": results}, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
