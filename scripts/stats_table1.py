#!/usr/bin/env python3
"""Stats for Table 1/2: Wilcoxon, bootstrap CIs, win-loss vs fair DDAE."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_completed(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("completed", data)


def per_dataset_pr(completed: dict, setting: str) -> pd.Series:
    rows = []
    for j in completed.values():
        if j.get("setting") != setting:
            continue
        m = j.get("metrics_mean") or j.get("metrics") or {}
        rows.append({"dataset": j["dataset"], "PR": float(m.get("PR-AUC", 0.0)) * 100.0})
    if not rows:
        return pd.Series(dtype=float)
    return pd.DataFrame(rows).groupby("dataset")["PR"].mean()


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    rng = np.random.RandomState(seed)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boots.append(sample.mean())
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return float(values.mean()), lo, hi


def wilcoxon_signed(deltas: np.ndarray) -> dict:
    try:
        from scipy.stats import wilcoxon

        if len(deltas) < 5 or np.allclose(deltas, 0):
            return {"statistic": None, "pvalue": None, "note": "insufficient or zero deltas"}
        stat, p = wilcoxon(deltas)
        return {"statistic": float(stat), "pvalue": float(p)}
    except Exception as e:
        return {"statistic": None, "pvalue": None, "note": str(e)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--completed", required=True)
    p.add_argument("--baseline", required=True, help="Fair DDAE completed.json")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    cpath = Path(args.completed)
    bpath = Path(args.baseline)
    if not cpath.is_absolute():
        cpath = PROJECT_ROOT / cpath
    if not bpath.is_absolute():
        bpath = PROJECT_ROOT / bpath
    out_dir = Path(args.out_dir) if args.out_dir else cpath.parent.parent / "thesis"
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cur = load_completed(cpath)
    base = load_completed(bpath)
    report = {}
    for setting in ["unsupervised", "semi-supervised"]:
        a = per_dataset_pr(cur, setting)
        b = per_dataset_pr(base, setting)
        merged = a.to_frame("ada").join(b.to_frame("ddae"), how="inner")
        if merged.empty:
            report[setting] = {"n": 0}
            continue
        merged["delta"] = merged["ada"] - merged["ddae"]
        wins = int((merged["delta"] > 1e-6).sum())
        ties = int((merged["delta"].abs() <= 1e-6).sum())
        losses = int((merged["delta"] < -1e-6).sum())
        mean, lo, hi = bootstrap_ci(merged["delta"].to_numpy())
        w = wilcoxon_signed(merged["delta"].to_numpy())
        report[setting] = {
            "n_datasets": int(len(merged)),
            "macro_ada_pr": float(merged["ada"].mean()),
            "macro_ddae_pr": float(merged["ddae"].mean()),
            "macro_delta_pr": float(merged["delta"].mean()),
            "bootstrap_ci95_delta": [lo, hi],
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "wilcoxon": w,
        }
        merged.reset_index().to_csv(out_dir / f"winloss_{setting}.csv", index=False)

    (out_dir / "stats_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
