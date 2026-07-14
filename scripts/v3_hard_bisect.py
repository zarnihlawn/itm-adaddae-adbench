#!/usr/bin/env python3
"""Hard-dataset ablation + semi bisect matrix for AdaDDAE v3 routing."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ablations import ABLATIONS, deep_update
from src.config import load_config
from src.data.datasets import build_registry
from src.eval.metrics import mean_std_metrics
from src.memory import cleanup_memory
from src.runlog.logger import RunLogger
from src.train.experiment import run_single_file

HARD_DATASETS = [
    "speech",
    "vowels",
    "letter",
    "Agnews",
    "20newsgroups",
    "ALOI",
    "Imdb",
    "SVHN",
    "cover",
    "celeba",
]

SEMI_BISECT_CANDIDATES = {
    "baseline_ddae": {
        "train": {"contrastive": False, "contrastive_alpha": 0.0, "hard_negative_mining": False},
        "adadae": {
            "use_danc": False,
            "use_scs": False,
            "use_ftp": False,
            "use_multiview": False,
            "use_uncertainty_view": False,
            "use_dte_view": False,
            "use_rejection_training": False,
        },
        "features": {"scaler": "standard", "pca_dim_threshold": 99999, "clip_outliers": False},
        "diffusion": {"num_timesteps": 50, "scheduler": "linear", "beta_end": 0.02},
    },
    "semi_cvnlp_ftp": {
        "train": {"contrastive": False, "contrastive_alpha": 0.0, "hard_negative_mining": False},
        "adadae": {
            "use_danc": False,
            "use_scs": False,
            "use_ftp": True,
            "use_multiview": False,
            "use_uncertainty_view": False,
            "use_dte_view": False,
            "use_rejection_training": False,
        },
        "features": {
            "scaler": "auto",
            "pca_dim_threshold": 128,
            "pca_max_components": 128,
            "pca_variance": 0.95,
            "clip_outliers": True,
            "clip_sigma": 5.0,
        },
        "diffusion": {"num_timesteps": 50, "scheduler": "linear", "beta_end": 0.02},
    },
    "semi_cvnlp_taps_004": {
        "train": {
            "contrastive": True,
            "contrastive_alpha": 0.04,
            "hard_negative_mining": True,
            "contrastive_adaptive_alpha": True,
        },
        "adadae": {
            "use_danc": False,
            "use_scs": False,
            "use_ftp": True,
            "use_multiview": False,
            "contrastive_pairing": "taps",
        },
        "features": {
            "scaler": "auto",
            "pca_dim_threshold": 128,
            "pca_max_components": 128,
            "pca_variance": 0.95,
            "clip_outliers": True,
            "clip_sigma": 5.0,
        },
        "diffusion": {"num_timesteps": 50, "scheduler": "linear", "beta_end": 0.02},
    },
    "semi_cvnlp_taps_006": {
        "train": {
            "contrastive": True,
            "contrastive_alpha": 0.06,
            "hard_negative_mining": True,
            "contrastive_adaptive_alpha": True,
        },
        "adadae": {
            "use_danc": False,
            "use_scs": False,
            "use_ftp": True,
            "use_multiview": False,
            "contrastive_pairing": "taps",
        },
        "features": {
            "scaler": "auto",
            "pca_dim_threshold": 128,
            "pca_max_components": 128,
            "pca_variance": 0.95,
            "clip_outliers": True,
            "clip_sigma": 5.0,
        },
        "diffusion": {"num_timesteps": 50, "scheduler": "linear", "beta_end": 0.02},
    },
    "semi_nlp_danc": {
        "train": {"contrastive": False, "contrastive_alpha": 0.0, "hard_negative_mining": False},
        "adadae": {
            "use_danc": True,
            "use_scs": False,
            "use_ftp": True,
            "use_multiview": False,
            "danc_contamination_mode": "label_free",
        },
        "features": {
            "scaler": "auto",
            "pca_dim_threshold": 128,
            "pca_max_components": 128,
            "pca_variance": 0.95,
            "clip_outliers": True,
            "clip_sigma": 5.0,
        },
    },
    "semi_speech_specialist": {
        "train": {"contrastive": False, "contrastive_alpha": 0.0, "hard_negative_mining": False},
        "adadae": {
            "use_danc": False,
            "use_scs": False,
            "use_ftp": True,
            "use_multiview": False,
            "use_rejection_training": True,
            "rejection_quantile": 0.95,
        },
        "features": {
            "scaler": "robust",
            "pca_dim_threshold": 99999,
            "clip_outliers": False,
        },
        "diffusion": {"num_timesteps": 80, "scheduler": "linear", "beta_end": 0.02},
    },
}

SEMI_TAIL_DATASETS = [
    "speech", "Imdb", "ALOI", "celeba", "Amazon", "Wilt", "SVHN", "Yelp",
    "20newsgroups", "CIFAR10", "Waveform", "census", "Agnews", "vertebral",
    "optdigits", "glass", "WPBC",
]

SEMI_TAIL_CANDIDATES = {
    "baseline_ddae": SEMI_BISECT_CANDIDATES["baseline_ddae"],
    "semi_nlp_danc": SEMI_BISECT_CANDIDATES["semi_nlp_danc"],
    "semi_speech_specialist": SEMI_BISECT_CANDIDATES["semi_speech_specialist"],
    "semi_rdt_tail": {
        "train": {"contrastive": False, "contrastive_alpha": 0.0, "hard_negative_mining": False},
        "adadae": {
            "use_danc": False,
            "use_scs": False,
            "use_ftp": True,
            "use_multiview": False,
            "use_rejection_training": True,
            "rejection_quantile": 0.90,
        },
        "features": {"scaler": "robust", "pca_dim_threshold": 99999, "clip_outliers": False},
        "diffusion": {"num_timesteps": 80, "scheduler": "linear", "beta_end": 0.02},
    },
    "semi_cvnlp_ftp": SEMI_BISECT_CANDIDATES["semi_cvnlp_ftp"],
    "semi_cvnlp_taps_004": SEMI_BISECT_CANDIDATES["semi_cvnlp_taps_004"],
    "ssts": ABLATIONS["ssts"],
    "taps": ABLATIONS["taps"],
    "lfdanc": ABLATIONS["lfdanc"],
    "ftp": ABLATIONS["ftp"],
    "rdt": ABLATIONS["rdt"],
}

UNSUP_BISECT_CANDIDATES = {
    "unsup_baseline": ABLATIONS["ddae_repro"],
    "unsup_lfdanc": ABLATIONS["lfdanc"],
    "unsup_ssts": ABLATIONS["ssts"],
}


def run_candidate(
    base_cfg: dict,
    candidate_name: str,
    overrides: dict,
    registry: list,
    setting: str,
    seed: int,
    logger: RunLogger,
) -> list[dict]:
    cfg = deep_update(base_cfg, overrides)
    rows = []
    for spec in registry:
        split_rows = []
        for rel in spec.relative_paths:
            if len(spec.relative_paths) > 1 and rel != spec.relative_paths[0]:
                continue
            row = run_single_file(
                npz_path=Path(base_cfg["paths"]["adbench_root"]) / rel,
                setting=setting,
                seed=seed,
                config=cfg,
                logger=logger,
                dataset_name=spec.name,
                split_name=rel,
                category=spec.category,
            )
            split_rows.append(row)
            cleanup_memory()
        agg = mean_std_metrics([r["metrics"] for r in split_rows])
        rows.append({
            "dataset": spec.name,
            "setting": setting,
            "candidate": candidate_name,
            "seed": seed,
            "PR-AUC": agg["PR-AUC"]["mean"] * 100,
            "ROC-AUC": agg["ROC-AUC"]["mean"] * 100,
            "category": spec.category,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Hard-dataset v3 bisect matrix")
    parser.add_argument("--config", default="configs/ablation_ladder.yaml")
    parser.add_argument("--hardware", default=None, help="Hardware profile; auto-CPU if no CUDA")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--datasets", nargs="*", default=HARD_DATASETS)
    parser.add_argument(
        "--setting",
        default=None,
        choices=["unsupervised", "semi-supervised"],
        help="Run one setting only",
    )
    parser.add_argument(
        "--full-tail",
        action="store_true",
        help="Run semi tail matrix on SEMI_TAIL_DATASETS with SEMI_TAIL_CANDIDATES",
    )
    parser.add_argument(
        "--steps",
        nargs="*",
        default=["ddae_repro", "lfdanc", "ssts", "taps", "oracle_danc"],
    )
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--skip-bisect", action="store_true")
    parser.add_argument("--out", default="results/thesis/v3_hard_dataset_matrix.csv")
    args = parser.parse_args()

    hardware = args.hardware
    if hardware is None:
        import torch

        hardware = "12gb" if torch.cuda.is_available() else None

    base = load_config(args.config, hardware=hardware)
    base["train"]["epochs"] = args.epochs
    base["train"]["eval_every"] = max(5, args.epochs // 5)
    base["paths"]["run_id"] = "v3_hard_bisect"

    adbench = Path(base["paths"]["adbench_root"])
    registry = build_registry(adbench)

    if args.full_tail:
        wanted = {d.lower() for d in SEMI_TAIL_DATASETS}
        registry = [s for s in registry if s.name.lower() in wanted]
        seeds = args.seeds or [111, 222, 333, 444, 555]
        settings = ["semi-supervised"]
        candidate_map = {"semi-supervised": SEMI_TAIL_CANDIDATES}
    else:
        wanted = {d.lower() for d in args.datasets}
        registry = [s for s in registry if s.name.lower() in wanted]
        seeds = args.seeds or [args.seed]
        settings = [args.setting] if args.setting else ["unsupervised", "semi-supervised"]
        candidate_map = {
            "unsupervised": UNSUP_BISECT_CANDIDATES,
            "semi-supervised": SEMI_BISECT_CANDIDATES,
        }

    if not registry:
        raise SystemExit(f"No datasets matched")

    results_dir = Path(base["paths"]["results_dir"])
    logger = RunLogger(results_dir / "logs" / "v3_hard_bisect.jsonl", run_id="v3_hard_bisect")
    all_rows: list[dict] = []

    if not args.skip_ablations and not args.full_tail:
        for setting in settings:
            for step in args.steps:
                if step not in ABLATIONS:
                    continue
                print(f"Ablation {step} / {setting}")
                for seed in seeds:
                    all_rows.extend(
                        run_candidate(
                            base, step, ABLATIONS[step], registry, setting, seed, logger
                        )
                    )

    if not args.skip_bisect:
        for setting in settings:
            candidates = candidate_map.get(setting, {})
            for name, overrides in candidates.items():
                for seed in seeds:
                    print(f"Bisect {name} / {setting} / seed {seed}")
                    all_rows.extend(
                        run_candidate(base, name, overrides, registry, setting, seed, logger)
                    )

    logger.close()
    df = pd.DataFrame(all_rows)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not df.empty:
        prior = pd.read_csv(out_path)
        df = pd.concat([prior, df], ignore_index=True)
        df = df.drop_duplicates(
            subset=["dataset", "setting", "candidate", "seed"], keep="last"
        )

    df.to_csv(out_path, index=False)

    best = (
        df.sort_values("PR-AUC", ascending=False)
        .groupby(["dataset", "setting"], as_index=False)
        .first()
    )
    best_path = out_path.with_name("v3_hard_dataset_best.csv")
    if args.full_tail:
        best_path = out_path.with_name("v31_semi_tail_best.csv")
    best.to_csv(best_path, index=False)

    print(f"\nWrote {out_path} ({len(df)} rows)")
    print(f"Wrote {best_path}")
    print("\nBest per dataset:")
    print(best[["dataset", "setting", "candidate", "PR-AUC"]].to_string(index=False))


if __name__ == "__main__":
    main()
