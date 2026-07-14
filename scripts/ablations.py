#!/usr/bin/env python3
"""
Ablation ladder for thesis (cumulative component isolation).

Steps: ddae_repro -> adadae_fixed -> ftp -> lfdanc -> ssts -> taps -> vus -> full_adadae
       oracle_danc (upper-bound ablation)
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.datasets import build_registry
from src.eval.metrics import mean_std_metrics
from src.memory import cleanup_memory
from src.runlog.logger import RunLogger
from src.train.experiment import run_single_file


ABLATIONS = {
    "ddae_repro": {
        "adadae": {
            "use_danc": False,
            "use_scs": False,
            "use_ftp": False,
            "use_multiview": False,
            "scs_max_timesteps": 50,
            "scs_mode": "full_sum",
            "scs_selection": "linspace",
            "contrastive_pairing": "random",
            "use_uncertainty_view": False,
            "fusion_mode": "fixed",
            "fusion_weights": {"reconstruction": 1.0, "latent": 0.0, "residual": 0.0, "uncertainty": 0.0},
        },
        "train": {"contrastive": False, "contrastive_alpha": 0.0},
        "features": {"scaler": "standard", "pca_dim_threshold": 99999, "clip_outliers": False},
        "diffusion": {"num_timesteps": 50, "scheduler": "linear", "time_emb_dim": 4},
    },
    "adadae_fixed": {
        "adadae": {
            "use_danc": False,
            "use_scs": True,
            "use_ftp": False,
            "use_multiview": True,
            "scs_mode": "full_sum",
            "scs_selection": "linspace",
            "scs_max_timesteps": 50,
            "contrastive_pairing": "random",
            "use_uncertainty_view": False,
            "fusion_mode": "fixed",
            "fusion_weights": {"reconstruction": 0.6, "latent": 0.3, "residual": 0.1, "uncertainty": 0.0},
        },
        "train": {"contrastive": False, "contrastive_alpha": 0.0},
        "features": {"scaler": "standard", "pca_dim_threshold": 99999, "clip_outliers": False},
        "diffusion": {"num_timesteps": 50, "scheduler": "linear", "time_emb_dim": 4},
    },
    "ftp": {
        "adadae": {
            "use_danc": False,
            "use_scs": False,
            "use_ftp": True,
            "use_multiview": False,
            "use_uncertainty_view": False,
            "contrastive_pairing": "random",
        },
        "train": {"contrastive": False, "contrastive_alpha": 0.0},
    },
    "lfdanc": {
        "adadae": {
            "use_danc": True,
            "use_scs": False,
            "use_ftp": True,
            "use_multiview": False,
            "danc_contamination_mode": "label_free",
            "use_uncertainty_view": False,
            "contrastive_pairing": "random",
        },
        "train": {"contrastive": False, "contrastive_alpha": 0.0},
    },
    "ssts": {
        "adadae": {
            "use_danc": True,
            "use_scs": True,
            "use_ftp": True,
            "use_multiview": False,
            "scs_mode": "snr_weighted",
            "scs_selection": "snr_stratified",
            "danc_contamination_mode": "label_free",
            "use_uncertainty_view": False,
            "use_dte_view": False,
            "use_rejection_training": False,
            "contrastive_pairing": "random",
        },
        "train": {"contrastive": False, "contrastive_alpha": 0.0},
    },
    "rdt": {
        "adadae": {
            "use_danc": True,
            "use_scs": True,
            "use_ftp": True,
            "use_multiview": False,
            "scs_mode": "snr_weighted",
            "scs_selection": "snr_stratified",
            "danc_contamination_mode": "label_free",
            "use_rejection_training": True,
            "rejection_quantile": 0.95,
            "use_dte_view": False,
            "use_uncertainty_view": False,
            "contrastive_pairing": "random",
        },
        "train": {"contrastive": False, "contrastive_alpha": 0.0},
    },
    "taps": {
        "adadae": {
            "use_danc": True,
            "use_scs": True,
            "use_ftp": True,
            "use_multiview": False,
            "scs_mode": "snr_weighted",
            "scs_selection": "snr_stratified",
            "danc_contamination_mode": "label_free",
            "contrastive_pairing": "taps",
            "use_uncertainty_view": False,
        },
        "train": {"contrastive": True, "contrastive_alpha": 0.15, "hard_negative_mining": True},
    },
    "vus": {
        "adadae": {
            "use_danc": True,
            "use_scs": True,
            "use_ftp": True,
            "use_multiview": True,
            "scs_mode": "snr_weighted",
            "scs_selection": "snr_stratified",
            "danc_contamination_mode": "label_free",
            "contrastive_pairing": "taps",
            "use_rejection_training": True,
            "use_uncertainty_view": True,
            "use_dte_view": False,
            "uncertainty_draws": 3,
            "fusion_mode": "calibrated",
            "fusion_weights": {
                "reconstruction": 0.45,
                "latent": 0.2,
                "residual": 0.1,
                "uncertainty": 0.15,
                "diffusion_time": 0.0,
            },
        },
        "train": {"contrastive": True, "contrastive_alpha": 0.15, "hard_negative_mining": True},
    },
    "dte": {
        "adadae": {
            "use_danc": True,
            "use_scs": True,
            "use_ftp": True,
            "use_multiview": True,
            "scs_mode": "snr_weighted",
            "scs_selection": "snr_stratified",
            "danc_contamination_mode": "label_free",
            "contrastive_pairing": "taps",
            "use_rejection_training": True,
            "use_uncertainty_view": True,
            "use_dte_view": True,
            "dte_knn": 5,
            "fusion_mode": "calibrated",
            "fusion_weights": {
                "reconstruction": 0.4,
                "latent": 0.2,
                "residual": 0.1,
                "uncertainty": 0.1,
                "diffusion_time": 0.2,
            },
        },
        "train": {"contrastive": True, "contrastive_alpha": 0.15, "hard_negative_mining": True},
    },
    "full_adadae": {
        "adadae": {
            "use_danc": True,
            "use_scs": True,
            "use_ftp": True,
            "use_multiview": True,
            "scs_max_timesteps": 32,
            "scs_mode": "snr_weighted",
            "scs_selection": "snr_stratified",
            "danc_contamination_mode": "label_free",
            "contrastive_pairing": "taps",
            "use_rejection_training": True,
            "use_uncertainty_view": True,
            "use_dte_view": True,
            "uncertainty_draws": 3,
            "dte_knn": 5,
            "fusion_mode": "calibrated",
            "fusion_weights": {
                "reconstruction": 0.4,
                "latent": 0.2,
                "residual": 0.1,
                "uncertainty": 0.1,
                "diffusion_time": 0.2,
            },
        },
        "train": {"contrastive": True, "contrastive_alpha": 0.15, "hard_negative_mining": True},
    },
    "oracle_danc": {
        "adadae": {
            "use_danc": True,
            "use_scs": True,
            "use_ftp": True,
            "use_multiview": True,
            "scs_mode": "snr_weighted",
            "scs_selection": "snr_stratified",
            "danc_contamination_mode": "oracle",
            "contrastive_pairing": "taps",
            "use_uncertainty_view": True,
            "fusion_mode": "calibrated",
        },
        "train": {"contrastive": True, "contrastive_alpha": 0.15},
    },
    # Legacy aliases
    "danc": {
        "adadae": {"use_danc": True, "use_scs": False, "use_ftp": True, "use_multiview": False},
        "train": {"contrastive": False, "contrastive_alpha": 0.0},
    },
    "scs": {
        "adadae": {"use_danc": True, "use_scs": True, "use_ftp": True, "use_multiview": False},
        "train": {"contrastive": False, "contrastive_alpha": 0.0},
    },
    "contrastive": {
        "adadae": {"use_danc": True, "use_scs": True, "use_ftp": True, "use_multiview": False},
        "train": {"contrastive": True, "contrastive_alpha": 0.2, "hard_negative_mining": True},
    },
}

LADDER_ORDER = [
    "ddae_repro",
    "adadae_fixed",
    "ftp",
    "lfdanc",
    "ssts",
    "rdt",
    "taps",
    "vus",
    "dte",
    "full_adadae",
    "oracle_danc",
]


def deep_update(base: dict, overrides: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ablation_ladder.yaml")
    parser.add_argument(
        "--hardware",
        type=str,
        default=None,
        help="Override hardware profile: 8gb|12gb|16gb|rtx5070ti",
    )
    parser.add_argument("--setting", default="semi-supervised")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--full", action="store_true")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["cardio", "thyroid", "breastw", "pendigits", "vowels"],
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--steps",
        nargs="*",
        default=None,
        help="Subset of ablation steps (default: full ladder)",
    )
    args = parser.parse_args()

    base = load_config(args.config, hardware=args.hardware)
    base["train"]["epochs"] = args.epochs
    base["train"]["eval_every"] = max(5, args.epochs // 5)
    base["paths"]["run_id"] = "ablations"

    adbench = Path(base["paths"]["adbench_root"])
    registry = build_registry(adbench)
    if not args.full:
        wanted = {d.lower() for d in args.datasets}
        registry = [s for s in registry if s.name.lower() in wanted]

    steps = args.steps or LADDER_ORDER
    results_dir = Path(base["paths"]["results_dir"])
    setting_slug = args.setting.replace("-", "_")
    logger = RunLogger(results_dir / "logs" / f"ablations_{setting_slug}.jsonl", run_id="ablations")
    out = {}

    for abl_name in steps:
        if abl_name not in ABLATIONS:
            print(f"Skip unknown step: {abl_name}")
            continue
        overrides = ABLATIONS[abl_name]
        cfg = deep_update(base, overrides)
        logger.info(f"Ablation {abl_name}", n_datasets=len(registry))
        ds_metrics = []
        for spec in registry:
            split_rows = []
            for rel in spec.relative_paths:
                if not args.full and len(spec.relative_paths) > 1 and rel != spec.relative_paths[0]:
                    continue
                row = run_single_file(
                    npz_path=adbench / rel,
                    setting=args.setting,
                    seed=args.seed,
                    config=cfg,
                    logger=logger,
                    dataset_name=spec.name,
                    split_name=rel,
                    category=spec.category,
                )
                split_rows.append(row)
                cleanup_memory()
            agg = mean_std_metrics([r["metrics"] for r in split_rows])
            ds_metrics.append({k: v["mean"] for k, v in agg.items()})
        overall = mean_std_metrics(ds_metrics)
        out[abl_name] = {
            "setting": args.setting,
            "seed": args.seed,
            "n_datasets": len(ds_metrics),
            "metrics": {k: v["mean"] for k, v in overall.items()},
        }
        print(abl_name, out[abl_name]["metrics"])

    out_dir = results_dir / "thesis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ablation_ladder_{setting_slug}.json"
    csv_path = out_dir / f"ablation_ladder_{setting_slug}.csv"
    # Merge with prior partial runs (same setting) so one-step loops do not erase history.
    if out_path.exists():
        try:
            prior = json.loads(out_path.read_text(encoding="utf-8"))
            if isinstance(prior, dict):
                prior.update(out)
                out = prior
        except json.JSONDecodeError:
            pass
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    rows = []
    baseline_pr = out.get("ddae_repro", {}).get("metrics", {}).get("PR-AUC", 0.0)
    for step in LADDER_ORDER:
        if step not in out:
            continue
        pr = out[step]["metrics"].get("PR-AUC", 0.0)
        rows.append({
            "step": step,
            "PR-AUC": pr * 100,
            "ROC-AUC": out[step]["metrics"].get("ROC-AUC", 0.0) * 100,
            "delta_PR_vs_ddae": (pr - baseline_pr) * 100,
        })
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    logger.close()
    print(f"\nWrote {out_path}")
    if not df.empty:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
