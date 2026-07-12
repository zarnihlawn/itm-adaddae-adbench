#!/usr/bin/env python3
"""Compare AdaDDAE results to published DDAE Table-1 means."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Published DDAE / DDAE-C means from arXiv:2508.00758 Table 1
PUBLISHED = {
    "unsupervised": {
        "DAE": {"PR-AUC": 19.85, "ROC-AUC": 63.73},
        "DDPM": {"PR-AUC": 30.09, "ROC-AUC": 70.01},
        "DTE-NP": {"PR-AUC": 29.47, "ROC-AUC": 72.15},
        "DDAE-C": {"PR-AUC": 31.97, "ROC-AUC": 73.72},
        "DDAE": {"PR-AUC": 32.77, "ROC-AUC": 74.08},
    },
    "semi-supervised": {
        "DAE": {"PR-AUC": 59.60, "ROC-AUC": 81.02},
        "DDPM": {"PR-AUC": 52.41, "ROC-AUC": 77.92},
        "DTE-NP": {"PR-AUC": 57.13, "ROC-AUC": 81.49},
        "DDAE-C": {"PR-AUC": 59.48, "ROC-AUC": 81.54},
        "DDAE": {"PR-AUC": 61.36, "ROC-AUC": 83.17},
    },
}


def load_completed(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate(completed: dict, setting: str):
    """Mean over datasets of (mean over seeds)."""
    by_ds = {}
    for key, row in completed.items():
        if row.get("setting") != setting:
            continue
        ds = row["dataset"]
        by_ds.setdefault(ds, []).append(row["metrics_mean"])

    if not by_ds:
        return None, None

    ds_means = []
    for ds, rows in by_ds.items():
        pr = np.mean([r["PR-AUC"] for r in rows])
        roc = np.mean([r["ROC-AUC"] for r in rows])
        ap = np.mean([r.get("AP", r["PR-AUC"]) for r in rows])
        # metrics are fractions 0-1; published Table 1 uses percentages
        ds_means.append({
            "dataset": ds,
            "PR-AUC": pr * 100.0,
            "ROC-AUC": roc * 100.0,
            "AP": ap * 100.0,
        })

    df = pd.DataFrame(ds_means)
    summary = {
        "PR-AUC": {"mean": float(df["PR-AUC"].mean()), "std": float(df["PR-AUC"].std(ddof=0))},
        "ROC-AUC": {"mean": float(df["ROC-AUC"].mean()), "std": float(df["ROC-AUC"].std(ddof=0))},
        "n_datasets": int(len(df)),
    }
    return summary, df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--completed",
        type=str,
        default="results/metrics/completed.json",
    )
    parser.add_argument("--out-dir", type=str, default="results/thesis")
    args = parser.parse_args()

    completed_path = Path(args.completed)
    if not completed_path.is_absolute():
        completed_path = PROJECT_ROOT / completed_path
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    state = load_completed(completed_path)
    completed = state.get("completed", state)

    rows = []
    for setting in ["unsupervised", "semi-supervised"]:
        summary, df = aggregate(completed, setting)
        if summary is None:
            print(f"No results for {setting}")
            continue
        if df is not None:
            df.to_csv(out_dir / f"per_dataset_{setting}.csv", index=False)

        ddae = PUBLISHED[setting]["DDAE"]
        row = {
            "setting": setting,
            "AdaDDAE_PR_AUC": summary["PR-AUC"]["mean"],
            "AdaDDAE_PR_std": summary["PR-AUC"]["std"],
            "AdaDDAE_ROC_AUC": summary["ROC-AUC"]["mean"],
            "AdaDDAE_ROC_std": summary["ROC-AUC"]["std"],
            "DDAE_PR_AUC": ddae["PR-AUC"],
            "DDAE_ROC_AUC": ddae["ROC-AUC"],
            "delta_PR": summary["PR-AUC"]["mean"] - ddae["PR-AUC"],
            "delta_ROC": summary["ROC-AUC"]["mean"] - ddae["ROC-AUC"],
            "n_datasets": summary["n_datasets"],
        }
        rows.append(row)
        print(f"\n=== {setting} ===")
        print(
            f"AdaDDAE  PR-AUC {summary['PR-AUC']['mean']:.2f}±{summary['PR-AUC']['std']:.2f}  "
            f"ROC-AUC {summary['ROC-AUC']['mean']:.2f}±{summary['ROC-AUC']['std']:.2f}"
        )
        print(f"DDAE     PR-AUC {ddae['PR-AUC']:.2f}  ROC-AUC {ddae['ROC-AUC']:.2f}")
        print(f"Δ        PR {row['delta_PR']:+.2f}  ROC {row['delta_ROC']:+.2f}")

    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "compare_to_ddae.csv", index=False)
        with open(out_dir / "compare_to_ddae.json", "w", encoding="utf-8") as f:
            json.dump({"published": PUBLISHED, "adadae": rows}, f, indent=2)
        print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
