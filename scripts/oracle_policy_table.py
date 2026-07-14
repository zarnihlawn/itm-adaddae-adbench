#!/usr/bin/env python3
"""Per-dataset best-policy oracle table for AdaDDAE v3 routing ceiling analysis."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import build_registry

PUBLISHED = {
    "unsupervised": {"PR-AUC": 32.77, "ROC-AUC": 74.08},
    "semi-supervised": {"PR-AUC": 61.36, "ROC-AUC": 83.17},
}

CV_NLP_NAMES = {
    "CIFAR10",
    "FashionMNIST",
    "MNIST-C",
    "MVTec-AD",
    "SVHN",
    "Agnews",
    "Amazon",
    "Imdb",
    "Yelp",
    "20newsgroups",
}

DEFAULT_SOURCES = {
    "backup_baseline": PROJECT_ROOT / "backup/ddae_baseline_570/metrics/completed.json",
    "unsup_ssts": PROJECT_ROOT / "results/adadae_unsup_ssts/metrics/completed.json",
    "semi_cvnlp": PROJECT_ROOT / "results/adadae_semi_cvnlp/metrics/completed.json",
}


def load_completed(path: Path) -> dict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("completed", data)


def aggregate_by_dataset(completed: dict, policy_name: str, setting: str | None = None) -> dict:
    """Mean PR/ROC per (dataset, setting) for a policy source."""
    buckets: dict[tuple[str, str], list[dict]] = {}
    for row in completed.values():
        ds = row.get("dataset", "")
        st = row.get("setting", "")
        if setting and st != setting:
            continue
        if not ds or not st:
            continue
        key = (ds, st)
        buckets.setdefault(key, []).append(row.get("metrics_mean", row.get("metrics", {})))

    out = {}
    for (ds, st), metrics_list in buckets.items():
        pr_vals = [m.get("PR-AUC", 0.0) * 100 for m in metrics_list if m]
        roc_vals = [m.get("ROC-AUC", 0.0) * 100 for m in metrics_list if m]
        if not pr_vals:
            continue
        out[(ds, st)] = {
            "policy": policy_name,
            "dataset": ds,
            "setting": st,
            "PR-AUC": sum(pr_vals) / len(pr_vals),
            "ROC-AUC": sum(roc_vals) / len(roc_vals) if roc_vals else 0.0,
            "n_seeds": len(pr_vals),
        }
    return out


def build_policy_rows(sources: dict[str, Path]) -> list[dict]:
    rows: list[dict] = []
    for policy_name, path in sources.items():
        completed = load_completed(path)
        if not completed:
            print(f"Skip missing/empty: {path}")
            continue
        for _, rec in aggregate_by_dataset(completed, policy_name).items():
            rows.append(rec)
    return rows


def oracle_table(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    idx_cols = ["dataset", "setting"]
    best = (
        df.sort_values("PR-AUC", ascending=False)
        .groupby(idx_cols, as_index=False)
        .first()
    )
    return best


def policy_win_counts(best: pd.DataFrame) -> pd.DataFrame:
    if best.empty:
        return pd.DataFrame()
    return (
        best.groupby(["setting", "policy"])
        .size()
        .reset_index(name="n_datasets")
        .sort_values(["setting", "n_datasets"], ascending=[True, False])
    )


def main():
    parser = argparse.ArgumentParser(description="Oracle best-policy ceiling table")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/thesis",
        help="Output directory for CSV/JSON",
    )
    parser.add_argument(
        "--backup",
        type=str,
        default=str(DEFAULT_SOURCES["backup_baseline"]),
    )
    parser.add_argument(
        "--unsup",
        type=str,
        default=str(DEFAULT_SOURCES["unsup_ssts"]),
    )
    parser.add_argument(
        "--semi-cvnlp",
        type=str,
        default=str(DEFAULT_SOURCES["semi_cvnlp"]),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "backup_baseline": Path(args.backup),
        "unsup_ssts": Path(args.unsup),
        "semi_cvnlp": Path(args.semi_cvnlp),
    }

    rows = build_policy_rows(sources)
    all_df = pd.DataFrame(rows)
    best_df = oracle_table(rows)

    # Category labels
    adbench = PROJECT_ROOT / "../ADBench/adbench/datasets"
    if adbench.exists():
        registry = build_registry(adbench.resolve())
        cat_map = {s.name: s.category for s in registry}
    else:
        cat_map = {n: "cv" if n in CV_NLP_NAMES and n not in {
            "Agnews", "Amazon", "Imdb", "Yelp", "20newsgroups"
        } else "nlp" if n in CV_NLP_NAMES else "classical" for n in set(best_df["dataset"])}

    best_df["category"] = best_df["dataset"].map(cat_map).fillna("classical")
    best_df["paper_PR"] = best_df["setting"].map(lambda s: PUBLISHED.get(s, {}).get("PR-AUC", 0.0))
    best_df["delta_PR_vs_paper"] = best_df["PR-AUC"] - best_df["paper_PR"]
    best_df["beats_paper"] = best_df["delta_PR_vs_paper"] > 0

    summary = {}
    for setting in ["unsupervised", "semi-supervised"]:
        sub = best_df[best_df["setting"] == setting]
        if sub.empty:
            continue
        pub = PUBLISHED[setting]["PR-AUC"]
        mean_pr = sub["PR-AUC"].mean()
        summary[setting] = {
            "oracle_mean_PR": float(mean_pr),
            "paper_PR": float(pub),
            "delta_vs_paper": float(mean_pr - pub),
            "beats_paper": bool(mean_pr > pub),
            "n_datasets": int(len(sub)),
            "n_beats_paper": int(sub["beats_paper"].sum()),
        }

    wins = policy_win_counts(best_df)

    all_path = out_dir / "oracle_policy_all.csv"
    best_path = out_dir / "oracle_policy_best.csv"
    wins_path = out_dir / "oracle_policy_wins.csv"
    summary_path = out_dir / "oracle_policy_summary.json"

    all_df.to_csv(all_path, index=False)
    best_df.to_csv(best_path, index=False)
    wins.to_csv(wins_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== Oracle policy ceiling ===")
    for setting, s in summary.items():
        flag = "PASS" if s["beats_paper"] else "FAIL"
        print(
            f"{setting}: oracle mean PR {s['oracle_mean_PR']:.2f}% "
            f"vs paper {s['paper_PR']:.2f}% "
            f"(delta {s['delta_vs_paper']:+.2f}%) [{flag}]"
        )
    print(f"\nPolicy wins:\n{wins.to_string(index=False)}")
    print(f"\nWrote {best_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
