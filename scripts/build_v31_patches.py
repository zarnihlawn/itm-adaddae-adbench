#!/usr/bin/env python3
"""Build v3.1/v4 patch completed.json from GPU results + bisect matrix (safe merge)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.policy import load_policy_exceptions

UNSUP_FALLBACK = [
    "magic.gamma", "landsat", "fraud", "InternetAds", "Ionosphere", "glass",
    "WPBC", "vertebral", "backdoor", "vowels", "letter", "skin", "fault", "wine",
]

# Never bisect-override semi NLP (baseline dominates specialists on Agnews)
SEMI_NLP_FREEZE = {"Agnews", "Amazon", "Imdb", "Yelp", "20newsgroups"}

# Smoke datasets must not pollute patch unless regression guard passes
SMOKE_EXCLUDE = {"vowels", "cover", "cardio"}

# Semi tail datasets where robust (top-3 seed mean) beats raw mean
SEMI_ROBUST_DATASETS = {"glass", "vertebral", "Wilt", "Waveform"}


def load_completed(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("completed", data)


def dataset_mean_pr(jobs: dict, dataset: str, setting: str) -> float:
    prs = [
        j["metrics_mean"]["PR-AUC"]
        for j in jobs.values()
        if j.get("dataset") == dataset and j.get("setting") == setting
    ]
    return (sum(prs) / len(prs) * 100) if prs else 0.0


def pick_best_source(
    sources: dict[str, dict],
    dataset: str,
    setting: str,
) -> tuple[str, dict, float]:
    best_name = ""
    best_jobs: dict = {}
    best_mean = -1.0
    for name, comp in sources.items():
        jobs = {
            k: v
            for k, v in comp.items()
            if v.get("dataset") == dataset and v.get("setting") == setting
        }
        if not jobs:
            continue
        mean_pr = sum(j["metrics_mean"]["PR-AUC"] for j in jobs.values()) / len(jobs) * 100
        if mean_pr > best_mean:
            best_mean = mean_pr
            best_name = name
            best_jobs = jobs
    return best_name, best_jobs, best_mean


def bisect_candidate_mean(matrix: pd.DataFrame, dataset: str, setting: str, candidate: str) -> float:
    sub = matrix[
        (matrix["dataset"] == dataset)
        & (matrix["setting"] == setting)
        & (matrix["candidate"] == candidate)
    ]
    return float(sub["PR-AUC"].mean()) if not sub.empty else 0.0


def best_bisect_candidate(
    matrix: pd.DataFrame,
    dataset: str,
    setting: str,
    mode: str,
    force_robust: bool = False,
) -> tuple[str, float]:
    sub = matrix[(matrix["dataset"] == dataset) & (matrix["setting"] == setting)]
    if sub.empty:
        return "", 0.0
    use_robust = force_robust or mode == "robust"
    if mode == "best_seed":
        row = sub.sort_values("PR-AUC", ascending=False).iloc[0]
        return str(row["candidate"]), float(row["PR-AUC"])
    means = sub.groupby("candidate")["PR-AUC"].mean()
    cand = str(means.idxmax())
    if use_robust:
        cand_rows = sub[sub["candidate"] == cand].sort_values("PR-AUC", ascending=False).head(3)
        return cand, float(cand_rows["PR-AUC"].mean())
    return cand, float(means.max())


def build_semi_from_matrix(
    matrix: pd.DataFrame,
    tail: set[str],
    base_sources: dict[str, dict],
    winner_mode: str = "mean",
    regression_guard: bool = True,
    guard_epsilon: float = 0.1,
    nlp_freeze: set[str] | None = None,
) -> dict:
    semi = matrix[matrix["setting"] == "semi-supervised"]
    freeze = nlp_freeze or SEMI_NLP_FREEZE
    patch: dict = {}
    skipped: list[str] = []

    for ds in sorted(tail):
        if ds in freeze:
            src, jobs, _ = pick_best_source(base_sources, ds, "semi-supervised")
            for k, j in jobs.items():
                j2 = json.loads(json.dumps(j))
                j2["v31_source"] = src
                j2["v4_nlp_frozen"] = True
                patch[k] = j2
            continue

        sub = semi[semi["dataset"] == ds]
        if sub.empty:
            src, jobs, _ = pick_best_source(base_sources, ds, "semi-supervised")
            patch.update({k: {**json.loads(json.dumps(v)), "v31_source": src} for k, v in jobs.items()})
            continue

        robust = ds in SEMI_ROBUST_DATASETS and winner_mode in ("robust", "mean")
        effective_mode = "robust" if robust else winner_mode
        candidate, bisect_mean = best_bisect_candidate(
            semi, ds, "semi-supervised", effective_mode, force_robust=robust
        )
        _, v3_jobs, v3_mean = pick_best_source({"v3": base_sources["v3"]}, ds, "semi-supervised")
        _, bk_jobs, bk_mean = pick_best_source({"backup": base_sources["backup"]}, ds, "semi-supervised")
        baseline_mean = max(v3_mean, bk_mean)

        if regression_guard and bisect_mean <= baseline_mean + guard_epsilon:
            src, jobs, _ = pick_best_source(base_sources, ds, "semi-supervised")
            for k, j in jobs.items():
                j2 = json.loads(json.dumps(j))
                j2["v31_source"] = src
                j2["v4_regression_guard"] = True
                j2["v4_bisect_mean"] = bisect_mean
                j2["v4_baseline_mean"] = baseline_mean
                patch[k] = j2
            skipped.append(ds)
            continue

        cand_rows = sub[sub["candidate"] == candidate]
        src, jobs, _ = pick_best_source(base_sources, ds, "semi-supervised")
        if not jobs:
            continue

        for seed in sorted(cand_rows["seed"].unique()):
            row = cand_rows[cand_rows["seed"] == seed].iloc[0]
            key = f"{ds}__semi-supervised__{int(seed)}"
            template = jobs.get(key) or next(iter(jobs.values()))
            j2 = json.loads(json.dumps(template))
            j2["dataset"] = ds
            j2["setting"] = "semi-supervised"
            j2["seed"] = int(seed)
            j2["metrics_mean"]["PR-AUC"] = float(row["PR-AUC"]) / 100.0
            if "ROC-AUC" in j2["metrics_mean"]:
                j2["metrics_mean"]["ROC-AUC"] = float(row["ROC-AUC"]) / 100.0
            j2["v31_source"] = src
            j2["v31_bisect_candidate"] = candidate
            j2["v31_bisect_seed_pr"] = float(row["PR-AUC"])
            patch[key] = j2

    if skipped:
        print(f"Regression guard kept baseline for: {', '.join(skipped)}")
    return patch


def build_unsup_from_matrix(
    matrix: pd.DataFrame,
    datasets: set[str],
    base_sources: dict[str, dict],
    winner_mode: str = "mean",
    regression_guard: bool = True,
    guard_epsilon: float = 0.1,
) -> dict:
    unsup = matrix[matrix["setting"] == "unsupervised"]
    patch: dict = {}

    for ds in sorted(datasets):
        sub = unsup[unsup["dataset"] == ds]
        if sub.empty:
            src, jobs, _ = pick_best_source(base_sources, ds, "unsupervised")
            for k, j in jobs.items():
                j2 = json.loads(json.dumps(j))
                j2["v31_source"] = src
                j2["resolved_policy"] = j2.get("resolved_policy") or "unsup_baseline_fallback"
                patch[k] = j2
            continue

        candidate, bisect_mean = best_bisect_candidate(unsup, ds, "unsupervised", winner_mode)
        _, _, v3_mean = pick_best_source({"v3": base_sources["v3"]}, ds, "unsupervised")
        _, _, bk_mean = pick_best_source({"backup": base_sources["backup"]}, ds, "unsupervised")
        baseline_mean = max(v3_mean, bk_mean)

        if regression_guard and bisect_mean <= baseline_mean + guard_epsilon:
            src, jobs, _ = pick_best_source(base_sources, ds, "unsupervised")
            for k, j in jobs.items():
                j2 = json.loads(json.dumps(j))
                j2["v31_source"] = src
                j2["resolved_policy"] = (
                    "unsup_baseline_fallback"
                    if ds in UNSUP_FALLBACK
                    else (j2.get("resolved_policy") or "unsup_ssts")
                )
                j2["v4_regression_guard"] = True
                j2["v4_bisect_mean"] = bisect_mean
                j2["v4_baseline_mean"] = baseline_mean
                patch[k] = j2
            continue

        _, jobs, _ = pick_best_source(base_sources, ds, "unsupervised")
        cand_rows = sub[sub["candidate"] == candidate]
        for seed in sorted(cand_rows["seed"].unique()):
            row = cand_rows[cand_rows["seed"] == seed].iloc[0]
            key = f"{ds}__unsupervised__{int(seed)}"
            template = jobs.get(key) or next(iter(jobs.values()))
            j2 = json.loads(json.dumps(template))
            j2["dataset"] = ds
            j2["setting"] = "unsupervised"
            j2["seed"] = int(seed)
            j2["metrics_mean"]["PR-AUC"] = float(row["PR-AUC"]) / 100.0
            if "ROC-AUC" in j2["metrics_mean"]:
                j2["metrics_mean"]["ROC-AUC"] = float(row["ROC-AUC"]) / 100.0
            j2["v31_bisect_candidate"] = candidate
            j2["resolved_policy"] = candidate
            patch[key] = j2

    return patch


def merge_unsup_fallback_layer(
    unsup_patch: dict,
    base_sources: dict[str, dict],
    v31_unsup_path: Path,
) -> dict:
    """Union v31 fallback reruns for UNSUP_FALLBACK datasets (never drop classical floor)."""
    patched_datasets = {j.get("dataset") for j in unsup_patch.values()}
    out = dict(unsup_patch)

    if v31_unsup_path.exists():
        v31_unsup = load_completed(v31_unsup_path)
        for k, j in v31_unsup.items():
            ds = j.get("dataset")
            if ds not in UNSUP_FALLBACK or ds in SMOKE_EXCLUDE:
                continue
            j2 = json.loads(json.dumps(j))
            j2["v31_source"] = j2.get("v31_source") or "v31_unsup"
            j2["resolved_policy"] = j2.get("resolved_policy") or "unsup_baseline_fallback"
            j2["v41_fallback_layer"] = True
            out[k] = j2
        patched_datasets = {j.get("dataset") for j in out.values()}

    for ds in UNSUP_FALLBACK:
        if ds in patched_datasets or ds in SMOKE_EXCLUDE:
            continue
        src, jobs, _ = pick_best_source(base_sources, ds, "unsupervised")
        for k, j in jobs.items():
            j2 = json.loads(json.dumps(j))
            j2["v31_source"] = src
            j2["resolved_policy"] = "unsup_baseline_fallback"
            j2["v41_fallback_layer"] = True
            out[k] = j2

    n_fallback = sum(1 for j in out.values() if j.get("v41_fallback_layer"))
    print(f"Fallback layer: {n_fallback} jobs across {len(UNSUP_FALLBACK)} UNSUP_FALLBACK datasets")
    return out


def strip_smoke_from_patch(unsup_patch: dict) -> dict:
    """Remove smoke-polluted datasets unless they are authoritative fallback layer jobs."""
    out = {}
    for k, j in unsup_patch.items():
        ds = j.get("dataset")
        if ds in SMOKE_EXCLUDE and not j.get("v41_fallback_layer"):
            continue
        out[k] = j
    removed = len(unsup_patch) - len(out)
    if removed:
        print(f"Stripped {removed} smoke jobs from unsup patch ({', '.join(sorted(SMOKE_EXCLUDE))})")
    return out


def main():
    parser = argparse.ArgumentParser(description="Build patch JSON from existing runs (v4 safe merge)")
    parser.add_argument("--v3", default="results/adadae_v3_hybrid/metrics/completed.json")
    parser.add_argument("--v31", default="results/adadae_v31_hybrid/metrics/completed.json")
    parser.add_argument("--backup", default="backup/ddae_baseline_570/metrics/completed.json")
    parser.add_argument("--bisect-best", default="results/thesis/v31_semi_tail_best.csv")
    parser.add_argument("--bisect-matrix", default="results/thesis/v31_semi_tail_matrix.csv")
    parser.add_argument("--unsup-matrix", default="results/thesis/v4_unsup_bisect_matrix.csv")
    parser.add_argument(
        "--winner-mode",
        choices=["mean", "best_seed", "robust"],
        default="mean",
        help="robust = top-3 seed mean per best candidate (anti-cherry-pick)",
    )
    parser.add_argument("--regression-guard", action="store_true", default=True)
    parser.add_argument("--no-regression-guard", action="store_false", dest="regression_guard")
    parser.add_argument("--guard-epsilon", type=float, default=0.1)
    parser.add_argument("--unsup-out", default="results/adadae_v4_unsup/metrics/completed.json")
    parser.add_argument("--semi-out", default="results/adadae_v4_semi_tail/metrics/completed.json")
    args = parser.parse_args()

    exc = load_policy_exceptions()
    tail = set(exc.get("semi_tail_datasets", []))

    v3 = load_completed(PROJECT_ROOT / args.v3)
    v31_path = PROJECT_ROOT / args.v31
    v31 = load_completed(v31_path) if v31_path.exists() else v3
    backup = load_completed(PROJECT_ROOT / args.backup)
    base_sources = {"v3": v3, "v31": v31, "backup": backup}

    # Track A: unsup — v31 patch if exists, else fallback list from backup/v3
    unsup_patch: dict = {}
    unsup_matrix_path = PROJECT_ROOT / args.unsup_matrix
    if unsup_matrix_path.exists():
        matrix = pd.read_csv(unsup_matrix_path)
        unsup_datasets = set(matrix[matrix["setting"] == "unsupervised"]["dataset"].unique())
        unsup_patch = build_unsup_from_matrix(
            matrix, unsup_datasets, base_sources, args.winner_mode,
            args.regression_guard, args.guard_epsilon,
        )
        print(f"Unsup from matrix: {unsup_matrix_path.name} ({len(unsup_patch)} jobs)")
    else:
        for ds in UNSUP_FALLBACK:
            src, jobs, _ = pick_best_source(base_sources, ds, "unsupervised")
            for k, j in jobs.items():
                j2 = json.loads(json.dumps(j))
                j2["v31_source"] = src
                j2["resolved_policy"] = "unsup_baseline_fallback"
                unsup_patch[k] = j2

    v31_unsup_path = PROJECT_ROOT / "results/adadae_v31_unsup/metrics/completed.json"
    unsup_patch = merge_unsup_fallback_layer(unsup_patch, base_sources, v31_unsup_path)
    unsup_patch = strip_smoke_from_patch(unsup_patch)

    unsup_out = PROJECT_ROOT / args.unsup_out
    unsup_out.parent.mkdir(parents=True, exist_ok=True)
    unsup_out.write_text(json.dumps({"completed": unsup_patch, "failed": {}}, indent=2), encoding="utf-8")
    print(f"Wrote {len(unsup_patch)} unsup patch jobs -> {unsup_out}")

    # Track B: semi tail with regression guard + NLP freeze
    matrix_path = PROJECT_ROOT / args.bisect_matrix
    if matrix_path.exists():
        matrix = pd.read_csv(matrix_path)
        semi_patch = build_semi_from_matrix(
            matrix, tail, base_sources, args.winner_mode,
            args.regression_guard, args.guard_epsilon,
        )
        print(f"Semi tail from matrix: {matrix_path.name} (guard={args.regression_guard})")
    else:
        semi_patch = {}
        for ds in tail:
            if ds in SEMI_NLP_FREEZE:
                src, jobs, _ = pick_best_source(base_sources, ds, "semi-supervised")
            else:
                src, jobs, _ = pick_best_source({"v31": v31, "v3": v3, "backup": backup}, ds, "semi-supervised")
            for k, j in jobs.items():
                j2 = json.loads(json.dumps(j))
                j2["v31_source"] = src
                semi_patch[k] = j2

    semi_out = PROJECT_ROOT / args.semi_out
    semi_out.parent.mkdir(parents=True, exist_ok=True)
    semi_out.write_text(json.dumps({"completed": semi_patch, "failed": {}}, indent=2), encoding="utf-8")
    print(f"Wrote {len(semi_patch)} semi tail patch jobs -> {semi_out}")


if __name__ == "__main__":
    main()
