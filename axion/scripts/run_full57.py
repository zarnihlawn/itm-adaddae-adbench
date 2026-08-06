#!/usr/bin/env python3
"""Full 57 × 2 × 5 protocol runner (Phase 4 — only after G2)."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axion.config import load_config  # noqa: E402
from axion.data.registry import build_registry  # noqa: E402
from axion.eval.metrics import PAPER_DDAE  # noqa: E402
from axion.train.experiment import aggregate_variants, run_dataset, save_job  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--model", default="axion")
    p.add_argument("--protocol", default="paper", choices=["paper", "integrity"])
    p.add_argument("--run-id", default="axion_full57")
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    adbench = Path(cfg["paths"]["adbench_root"])
    results_root = Path(cfg["paths"]["results_dir"]) / args.run_id
    metrics_dir = results_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    harness = cfg.get("harness", {})
    axion_cfg = dict(cfg.get("axion", {}) or {})
    max_train = (
        args.max_train_samples
        if args.max_train_samples is not None
        else int(harness.get("max_train_samples", 20000))
    )
    seeds = args.seeds or list(cfg.get("seeds", [111, 222, 333, 444, 555]))
    settings = ["unsupervised", "semi-supervised"]
    specs = build_registry(adbench)

    by_setting: dict[str, list[dict]] = defaultdict(list)
    completed = 0
    total = len(specs) * len(settings) * len(seeds)

    for spec in specs:
        for setting in settings:
            seed_aggs = []
            for seed in seeds:
                variant_results = run_dataset(
                    spec,
                    adbench_root=adbench,
                    setting=setting,
                    seed=seed,
                    model_name=args.model,
                    model_kwargs=axion_cfg if args.model == "axion" else None,
                    protocol=args.protocol,
                    val_fraction=float(cfg.get("integrity", {}).get("val_fraction", 0.2)),
                    max_train_samples=max_train,
                    max_variants=0,
                )
                for r in variant_results:
                    save_job(r, metrics_dir)
                agg = aggregate_variants(variant_results)
                seed_aggs.append(agg)
                completed += 1
                print(
                    f"[{completed}/{total}] {spec.name:16s} {setting:18s} seed={seed} "
                    f"PR={agg['PR-AUC']:.2f} ROC={agg['ROC-AUC']:.2f}",
                    flush=True,
                )
            row = {
                "dataset": spec.name,
                "setting": setting,
                "PR-AUC": float(np.mean([a["PR-AUC"] for a in seed_aggs])),
                "ROC-AUC": float(np.mean([a["ROC-AUC"] for a in seed_aggs])),
            }
            by_setting[setting].append(row)

    summary = {
        "time": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "protocol": args.protocol,
        "paper": PAPER_DDAE,
        "macro": {},
        "per_dataset": {s: rows for s, rows in by_setting.items()},
    }
    for setting, rows in by_setting.items():
        pr = float(np.mean([r["PR-AUC"] for r in rows]))
        roc = float(np.mean([r["ROC-AUC"] for r in rows]))
        paper = PAPER_DDAE[setting]
        summary["macro"][setting] = {
            "PR-AUC": pr,
            "ROC-AUC": roc,
            "delta_PR_vs_paper": pr - paper["PR-AUC"],
            "delta_ROC_vs_paper": roc - paper["ROC-AUC"],
            "n_datasets": len(rows),
            "pass_paper": pr > paper["PR-AUC"] and roc > paper["ROC-AUC"],
        }
        print(
            f"MACRO {setting}: PR={pr:.2f} ROC={roc:.2f} "
            f"ΔPR={pr - paper['PR-AUC']:+.2f} ΔROC={roc - paper['ROC-AUC']:+.2f}",
            flush=True,
        )

    out = results_root / "compare_to_ddae.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
