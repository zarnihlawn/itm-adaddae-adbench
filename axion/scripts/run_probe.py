#!/usr/bin/env python3
"""Run Phase-1 probe harness (AnoDDAE protocol).

Examples:
  python scripts/run_probe.py --smoke
  python scripts/run_probe.py --datasets breastw cardio --seeds 111 --settings semi-supervised
  python scripts/run_probe.py --all-probe
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axion.config import load_config  # noqa: E402
from axion.data.registry import registry_by_name  # noqa: E402
from axion.eval.metrics import PAPER_DDAE  # noqa: E402
from axion.train.experiment import (  # noqa: E402
    aggregate_variants,
    run_dataset,
    save_job,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AXION Phase-1 probe harness")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--smoke", action="store_true", help="1 dataset × 1 seed × both settings")
    p.add_argument("--all-probe", action="store_true", help="Full probe list from config")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument(
        "--settings",
        nargs="*",
        default=["unsupervised", "semi-supervised"],
        choices=["unsupervised", "semi-supervised"],
    )
    p.add_argument("--model", default="axion")
    p.add_argument("--protocol", default="paper", choices=["paper", "integrity"])
    p.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Cap train rows (default from config harness.max_train_samples)",
    )
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-variants", type=int, default=None, help="Cap CV/NLP variants (0=all)")
    p.add_argument("--run-id", default="axion_probe")
    p.add_argument("--loop-log", action="store_true", help="Append macros to probe_loop.jsonl")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    adbench = Path(cfg["paths"]["adbench_root"])
    results_root = Path(cfg["paths"]["results_dir"]) / args.run_id
    metrics_dir = results_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    probe_cfg = cfg.get("probe", {})
    harness = cfg.get("harness", {})
    axion_cfg = dict(cfg.get("axion", {}) or {})
    max_train = (
        args.max_train_samples
        if args.max_train_samples is not None
        else int(harness.get("max_train_samples", 20000))
    )
    if args.epochs is not None:
        axion_cfg["epochs"] = args.epochs
    val_fraction = float(cfg.get("integrity", {}).get("val_fraction", 0.2))
    model_name = args.model or harness.get("model", "axion")
    max_variants = (
        args.max_variants
        if args.max_variants is not None
        else int(harness.get("max_variants", 0))
    )

    if args.smoke:
        datasets = ["breastw"]
        seeds = [111]
        axion_cfg.setdefault("epochs", 30)
    elif args.all_probe:
        datasets = list(probe_cfg.get("datasets", []))
        seeds = list(probe_cfg.get("seeds", [111, 222, 333]))
    else:
        datasets = args.datasets or ["breastw", "cardio"]
        seeds = args.seeds or [111]

    reg = registry_by_name(adbench)
    missing = [d for d in datasets if d not in reg]
    if missing:
        raise SystemExit(f"Unknown datasets: {missing}")

    summary_rows = []
    by_setting: dict[str, list[dict]] = defaultdict(list)

    for ds in datasets:
        spec = reg[ds]
        for setting in args.settings:
            seed_aggs = []
            for seed in seeds:
                variant_results = run_dataset(
                    spec,
                    adbench_root=adbench,
                    setting=setting,
                    seed=seed,
                    model_name=model_name,
                    model_kwargs=axion_cfg if model_name == "axion" else None,
                    protocol=args.protocol,
                    val_fraction=val_fraction,
                    max_train_samples=max_train,
                    max_variants=max_variants,
                )
                for r in variant_results:
                    save_job(r, metrics_dir)
                agg = aggregate_variants(variant_results)
                seed_aggs.append(agg)
                print(
                    f"{ds:16s} {setting:18s} seed={seed} "
                    f"PR={agg['PR-AUC']:.2f} ROC={agg['ROC-AUC']:.2f} "
                    f"vars={int(agg['n_variants'])} "
                    f"t={sum(r.seconds for r in variant_results):.1f}s",
                    flush=True,
                )
            macro = {
                "dataset": ds,
                "setting": setting,
                "PR-AUC": float(np.mean([a["PR-AUC"] for a in seed_aggs])),
                "ROC-AUC": float(np.mean([a["ROC-AUC"] for a in seed_aggs])),
                "n_seeds": len(seeds),
            }
            summary_rows.append(macro)
            by_setting[setting].append(macro)

    compare = {
        "model": model_name,
        "protocol": args.protocol,
        "paper": PAPER_DDAE,
        "probe_macro": {},
        "per_dataset": summary_rows,
        "axion": axion_cfg,
    }
    for setting, rows in by_setting.items():
        pr = float(np.mean([r["PR-AUC"] for r in rows]))
        roc = float(np.mean([r["ROC-AUC"] for r in rows]))
        paper = PAPER_DDAE[setting]
        compare["probe_macro"][setting] = {
            "PR-AUC": pr,
            "ROC-AUC": roc,
            "delta_PR_vs_paper": pr - paper["PR-AUC"],
            "delta_ROC_vs_paper": roc - paper["ROC-AUC"],
            "n_datasets": len(rows),
            "pass_probe_margin": (pr >= paper["PR-AUC"] + 2.0)
            and (roc >= paper["ROC-AUC"] + 1.0),
        }

    out_json = results_root / "probe_summary.json"
    out_json.write_text(json.dumps(compare, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}")
    for setting, m in compare["probe_macro"].items():
        paper = PAPER_DDAE[setting]
        flag = "PASS" if m["pass_probe_margin"] else "FAIL"
        print(
            f"MACRO {setting} [{flag}]: PR={m['PR-AUC']:.2f} (paper {paper['PR-AUC']}, "
            f"Δ={m['delta_PR_vs_paper']:+.2f})  "
            f"ROC={m['ROC-AUC']:.2f} (paper {paper['ROC-AUC']}, "
            f"Δ={m['delta_ROC_vs_paper']:+.2f})"
        )

    if args.loop_log:
        from datetime import datetime, timezone

        loop_path = results_root / "probe_loop.jsonl"
        entry = {
            "time": datetime.now(timezone.utc).isoformat(),
            "model": model_name,
            "axion": axion_cfg,
            "probe_macro": compare["probe_macro"],
            "datasets": datasets,
            "seeds": seeds,
        }
        with open(loop_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"Appended {loop_path}")


if __name__ == "__main__":
    main()
