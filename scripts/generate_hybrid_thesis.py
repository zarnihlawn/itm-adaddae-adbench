#!/usr/bin/env python3
"""Generate hybrid thesis tables: Table-1, ablation waterfall, negative-result note."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

BACKUP_COMPARE = PROJECT_ROOT / "backup" / "ddae_baseline_570" / "thesis" / "compare_to_ddae.csv"
ABLATION_SEMI = PROJECT_ROOT / "results" / "thesis" / "ablation_ladder_semi_supervised.csv"
ABLATION_UNSUP = PROJECT_ROOT / "results" / "thesis" / "ablation_ladder_unsupervised.csv"
ORACLE_SUMMARY = PROJECT_ROOT / "results" / "thesis" / "oracle_policy_summary.json"
V3_HARD_BEST = PROJECT_ROOT / "results" / "thesis" / "v3_hard_dataset_best.csv"

PUBLISHED = {
    "unsupervised": {"PR-AUC": 32.77, "ROC-AUC": 74.08},
    "semi-supervised": {"PR-AUC": 61.36, "ROC-AUC": 83.17},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hybrid-dir",
        default="results/adadae_v2_hybrid",
        help="Hybrid results root (contains thesis/compare_to_ddae.csv)",
    )
    args = parser.parse_args()

    hybrid_root = Path(args.hybrid_dir)
    if not hybrid_root.is_absolute():
        hybrid_root = PROJECT_ROOT / hybrid_root
    out_dir = hybrid_root / "thesis"
    hybrid_compare = out_dir / "compare_to_ddae.csv"
    model_label = hybrid_root.name.replace("adadae_", "AdaDDAE_")

    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    if BACKUP_COMPARE.exists():
        backup_df = pd.read_csv(BACKUP_COMPARE)
        backup_df["model"] = "DDAE_baseline_570"
        rows.append(backup_df)

    if hybrid_compare.exists():
        hybrid_df = pd.read_csv(hybrid_compare)
        hybrid_df["model"] = model_label
        rows.append(hybrid_df)

    if rows:
        table1 = pd.concat(rows, ignore_index=True)
        table1_path = out_dir / "table1_baseline_vs_hybrid.csv"
        table1.to_csv(table1_path, index=False)

        lines = [f"# {model_label} vs DDAE (Table 1)", ""]
        for _, r in table1.iterrows():
            setting = r["setting"]
            pub = PUBLISHED.get(setting, {})
            lines.append(f"## {setting} ({r.get('model', 'model')})")
            lines.append(
                f"- PR-AUC: {r['AdaDDAE_PR_AUC']:.2f}% vs DDAE paper {pub.get('PR-AUC', '?')}% "
                f"(delta {r.get('delta_PR', float('nan')):+.2f})"
            )
            lines.append(
                f"- ROC-AUC: {r['AdaDDAE_ROC_AUC']:.2f}% vs DDAE paper {pub.get('ROC-AUC', '?')}% "
                f"(delta {r.get('delta_ROC', float('nan')):+.2f})"
            )
            lines.append("")
        (out_dir / "table1_summary.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote {table1_path}")

    waterfall_parts = []
    for path, label in [(ABLATION_SEMI, "semi"), (ABLATION_UNSUP, "unsup")]:
        if path.exists():
            df = pd.read_csv(path)
            df["setting_group"] = label
            waterfall_parts.append(df)
    if waterfall_parts:
        wf = pd.concat(waterfall_parts, ignore_index=True)
        wf.to_csv(out_dir / "ablation_waterfall.csv", index=False)
        print(f"Wrote {out_dir / 'ablation_waterfall.csv'}")

    negative = {
        "finding": "Monolithic default_gpu / vus stack regresses on GPU",
        "evidence_semi_vus_pr_delta": -18.7,
        "evidence_file": str(ABLATION_SEMI),
        "recommendation": "Use v3 dataset-aware routing; ban VUS on semi; revert NLP semi to baseline",
    }
    with open(out_dir / "negative_result_default_gpu.json", "w", encoding="utf-8") as f:
        json.dump(negative, f, indent=2)

    attribution = {
        "version": "v3" if "v3" in hybrid_root.name else "v2",
        "unsupervised_default": ["LF-DANC", "MANS", "SSTS", "FTP"],
        "unsup_baseline_fallback": [
            "vowels",
            "letter",
            "skin",
            "fault",
            "wine",
            "glass",
        ],
        "unsup_nlp_baseline": ["Agnews", "Amazon", "Imdb", "Yelp", "20newsgroups"],
        "semi_classical": ["DDAE-faithful baseline (no AdaDDAE extras)"],
        "semi_cv": ["FTP", "fixed T=50 linear", "contrastive off"],
        "semi_nlp": ["DDAE-faithful baseline (FTP+TAPS regressed on Agnews/20news)"],
        "semi_specialists": {"speech": "RobustScaler + FTP + RDT + T=80"},
        "banned_components": ["VUS", "DTE-View", "RDT on semi classical", "calibrated fusion"],
        "evidence_files": {
            "oracle_summary": str(ORACLE_SUMMARY),
            "v3_hard_best": str(V3_HARD_BEST),
        },
    }
    if ORACLE_SUMMARY.exists():
        attribution["oracle_ceiling"] = json.loads(ORACLE_SUMMARY.read_text(encoding="utf-8"))
    with open(out_dir / "component_attribution.json", "w", encoding="utf-8") as f:
        json.dump(attribution, f, indent=2)

    print(f"Thesis artifacts in {out_dir}")


if __name__ == "__main__":
    main()
