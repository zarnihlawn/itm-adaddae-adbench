#!/usr/bin/env python3
"""Generate hybrid thesis tables: Table-1, ablation waterfall, negative-result note."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "results" / "adadae_v2_hybrid" / "thesis"
BACKUP_COMPARE = PROJECT_ROOT / "backup" / "ddae_baseline_570" / "thesis" / "compare_to_ddae.csv"
HYBRID_COMPARE = OUT_DIR / "compare_to_ddae.csv"
ABLATION_SEMI = PROJECT_ROOT / "results" / "thesis" / "ablation_ladder_semi_supervised.csv"
ABLATION_UNSUP = PROJECT_ROOT / "results" / "thesis" / "ablation_ladder_unsupervised.csv"

PUBLISHED = {
    "unsupervised": {"PR-AUC": 32.77, "ROC-AUC": 74.08},
    "semi-supervised": {"PR-AUC": 61.36, "ROC-AUC": 83.17},
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    if BACKUP_COMPARE.exists():
        backup_df = pd.read_csv(BACKUP_COMPARE)
        backup_df["model"] = "DDAE_baseline_570"
        rows.append(backup_df)

    if HYBRID_COMPARE.exists():
        hybrid_df = pd.read_csv(HYBRID_COMPARE)
        hybrid_df["model"] = "AdaDDAE_v2_hybrid"
        rows.append(hybrid_df)

    if rows:
        table1 = pd.concat(rows, ignore_index=True)
        table1_path = OUT_DIR / "table1_baseline_vs_hybrid.csv"
        table1.to_csv(table1_path, index=False)

        lines = ["# AdaDDAE v2 Hybrid vs DDAE (Table 1)", ""]
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
        (OUT_DIR / "table1_summary.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote {table1_path}")

    waterfall_parts = []
    for path, label in [(ABLATION_SEMI, "semi"), (ABLATION_UNSUP, "unsup")]:
        if path.exists():
            df = pd.read_csv(path)
            df["setting_group"] = label
            waterfall_parts.append(df)
    if waterfall_parts:
        wf = pd.concat(waterfall_parts, ignore_index=True)
        wf.to_csv(OUT_DIR / "ablation_waterfall.csv", index=False)
        print(f"Wrote {OUT_DIR / 'ablation_waterfall.csv'}")

    negative = {
        "finding": "Monolithic default_gpu / vus stack regresses on GPU",
        "evidence_semi_vus_pr_delta": -18.7,
        "evidence_file": str(ABLATION_SEMI),
        "recommendation": "Use setting-aware routing: unsup_ssts + semi_cvnlp FTP/TAPS + classical baselines_ddae",
    }
    with open(OUT_DIR / "negative_result_default_gpu.json", "w", encoding="utf-8") as f:
        json.dump(negative, f, indent=2)

    attribution = {
        "unsupervised": ["LF-DANC", "MANS", "SSTS", "FTP"],
        "semi_classical": ["DDAE-faithful baseline (no AdaDDAE extras)"],
        "semi_cv_nlp": ["FTP", "TAPS (contrastive_alpha=0.06)", "fixed T=50 linear"],
        "banned_components": ["VUS", "DTE-View", "RDT", "calibrated fusion"],
    }
    with open(OUT_DIR / "component_attribution.json", "w", encoding="utf-8") as f:
        json.dump(attribution, f, indent=2)

    print(f"Thesis artifacts in {OUT_DIR}")


if __name__ == "__main__":
    main()
