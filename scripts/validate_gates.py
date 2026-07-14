#!/usr/bin/env python3
"""Validate AdaDDAE v3 Table-1 gates vs DDAE paper and backup."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PUBLISHED = {
    "unsupervised": {"PR-AUC": 32.77, "ROC-AUC": 74.08},
    "semi-supervised": {"PR-AUC": 61.36, "ROC-AUC": 83.17},
}


def load_completed(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("completed", data)


def mean_metrics(completed: dict, setting: str) -> dict:
    prs, rocs = [], []
    for job in completed.values():
        if job.get("setting") != setting:
            continue
        m = job.get("metrics_mean", job.get("metrics", {}))
        prs.append(m.get("PR-AUC", 0.0) * 100)
        rocs.append(m.get("ROC-AUC", 0.0) * 100)
    return {
        "PR-AUC": sum(prs) / len(prs) if prs else 0.0,
        "ROC-AUC": sum(rocs) / len(rocs) if rocs else 0.0,
        "n_jobs": len(prs),
    }


def per_dataset_loss_vs_ref(hybrid: dict, ref: dict, setting: str, threshold: float = 0.5) -> tuple[int, list[str]]:
    import pandas as pd

    def agg(completed):
        rows = []
        for j in completed.values():
            if j.get("setting") != setting:
                continue
            rows.append({"dataset": j["dataset"], "PR": j["metrics_mean"]["PR-AUC"] * 100})
        return pd.DataFrame(rows).groupby("dataset")["PR"].mean()

    h = agg(hybrid)
    r = agg(ref)
    merged = h.to_frame("h").join(r.to_frame("r"), how="inner")
    merged["delta"] = merged["h"] - merged["r"]
    losers = merged[merged["delta"] < -threshold].index.tolist()
    return len(losers), losers


def per_dataset_backup_loss(hybrid: dict, backup: dict, setting: str, threshold: float = 2.0) -> int:
    n, _ = per_dataset_loss_vs_ref(hybrid, backup, setting, threshold)
    return n


def main():
    parser = argparse.ArgumentParser(description="Validate v3 gates")
    parser.add_argument("--completed", required=True)
    parser.add_argument(
        "--backup",
        default="backup/ddae_baseline_570/metrics/completed.json",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--v31-ref",
        default="results/adadae_v31_hybrid/metrics/completed.json",
        help="Reference hybrid for G6 no-regression vs v3.1",
    )
    args = parser.parse_args()

    completed_path = Path(args.completed)
    if not completed_path.is_absolute():
        completed_path = PROJECT_ROOT / completed_path
    backup_path = Path(args.backup)
    if not backup_path.is_absolute():
        backup_path = PROJECT_ROOT / backup_path

    hybrid = load_completed(completed_path)
    backup = load_completed(backup_path)

    gates = {}
    all_pass = True
    for setting in ["unsupervised", "semi-supervised"]:
        m = mean_metrics(hybrid, setting)
        pub = PUBLISHED[setting]
        g1 = m["PR-AUC"] >= pub["PR-AUC"]
        g3 = m["ROC-AUC"] >= pub["ROC-AUC"]
        gates[setting] = {
            "PR-AUC": m["PR-AUC"],
            "ROC-AUC": m["ROC-AUC"],
            "delta_PR_vs_paper": m["PR-AUC"] - pub["PR-AUC"],
            "delta_ROC_vs_paper": m["ROC-AUC"] - pub["ROC-AUC"],
            "G_PR_beats_paper": g1,
            "G_ROC_beats_paper": g3,
        }
        all_pass = all_pass and g1 and g3

    g4_unsup = per_dataset_backup_loss(hybrid, backup, "unsupervised")
    g4_semi = per_dataset_backup_loss(hybrid, backup, "semi-supervised")
    gates["G4_backup_regressions"] = {
        "unsup_datasets_losing_gt_2pct": g4_unsup,
        "semi_datasets_losing_gt_2pct": g4_semi,
        "pass": g4_unsup <= 5 and g4_semi <= 5,
    }
    all_pass = all_pass and gates["G4_backup_regressions"]["pass"]

    # G6: no dataset loses >0.5% PR vs v3.1 reference
    v31_path = Path(args.v31_ref)
    if not v31_path.is_absolute():
        v31_path = PROJECT_ROOT / v31_path
    g6_pass = True
    g6_detail: dict = {}
    if v31_path.exists():
        v31 = load_completed(v31_path)
        for setting in ["unsupervised", "semi-supervised"]:
            n_loss, losers = per_dataset_loss_vs_ref(hybrid, v31, setting, threshold=0.5)
            g6_detail[setting] = {"n_losing_gt_0_5pct": n_loss, "datasets": losers}
            if n_loss > 0:
                g6_pass = False
    gates["G6_vs_v31_regressions"] = {
        **g6_detail,
        "pass": g6_pass,
        "ref": str(v31_path),
    }
    all_pass = all_pass and g6_pass

    gates["all_pass"] = all_pass
    gates["n_jobs"] = len(hybrid)

    print("=== Validation gates ===")
    for setting in ["unsupervised", "semi-supervised"]:
        g = gates[setting]
        print(
            f"{setting}: PR {g['PR-AUC']:.2f}% (delta {g['delta_PR_vs_paper']:+.2f}) "
            f"{'PASS' if g['G_PR_beats_paper'] else 'FAIL'}"
        )
        print(
            f"  ROC {g['ROC-AUC']:.2f}% (delta {g['delta_ROC_vs_paper']:+.2f}) "
            f"{'PASS' if g['G_ROC_beats_paper'] else 'FAIL'}"
        )
    print(
        f"G4 backup regressions: unsup {g4_unsup}, semi {g4_semi} "
        f"{'PASS' if gates['G4_backup_regressions']['pass'] else 'FAIL'}"
    )
    if "G6_vs_v31_regressions" in gates:
        g6 = gates["G6_vs_v31_regressions"]
        for setting in ["unsupervised", "semi-supervised"]:
            if setting in g6:
                print(
                    f"G6 vs v3.1 {setting}: {g6[setting]['n_losing_gt_0_5pct']} datasets "
                    f"{'PASS' if g6['pass'] else 'FAIL'}"
                )
                if g6[setting]["datasets"]:
                    print(f"  losers: {g6[setting]['datasets']}")
        print(f"G6 overall: {'PASS' if g6['pass'] else 'FAIL'}")
    print(f"ALL PASS: {all_pass}")

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(gates, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
