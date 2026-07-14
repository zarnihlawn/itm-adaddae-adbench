#!/usr/bin/env python3
"""Merge hybrid AdaDDAE v2 completed.json from backup + partial runs."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import build_registry

# CV/NLP display names (matches build_registry)
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


def load_state(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "completed" in data:
        return data
    return {"completed": data, "failed": {}}


def filter_completed(state: dict, setting: str | None = None, datasets: set[str] | None = None) -> dict:
    out = {}
    for key, row in state.get("completed", {}).items():
        if setting and row.get("setting") != setting:
            continue
        if datasets is not None and row.get("dataset") not in datasets:
            continue
        out[key] = row
    return out


def copy_metrics(src_dir: Path, dst_dir: Path, keys: set[str]) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for key in keys:
        src = src_dir / f"{key}.json"
        if src.exists():
            shutil.copy2(src, dst_dir / f"{key}.json")
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser(description="Merge hybrid AdaDDAE v2/v3 results")
    parser.add_argument(
        "--semi-classical",
        type=str,
        required=True,
        help="completed.json with classical semi jobs (e.g. backup baseline)",
    )
    parser.add_argument(
        "--semi-cvnlp",
        type=str,
        required=True,
        help="completed.json with CV/NLP semi jobs",
    )
    parser.add_argument(
        "--unsup",
        type=str,
        required=True,
        help="completed.json with all unsupervised jobs",
    )
    parser.add_argument(
        "--semi-cvnlp-source",
        type=str,
        choices=["path", "backup"],
        default="path",
        help="Use --semi-cvnlp path (path) or --semi-classical backup for CV/NLP semi (backup)",
    )
    parser.add_argument(
        "--patch",
        type=str,
        default=None,
        help="Optional completed.json whose jobs override same keys (v3 selective reruns)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/adadae_v2_hybrid/metrics/completed.json",
    )
    parser.add_argument(
        "--copy-metrics",
        action="store_true",
        help="Copy per-job JSON files into output metrics dir",
    )
    args = parser.parse_args()

    semi_classical_path = Path(args.semi_classical)
    semi_cvnlp_path = Path(args.semi_cvnlp)
    unsup_path = Path(args.unsup)
    patch_path = Path(args.patch) if args.patch else None
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path

    adbench = PROJECT_ROOT / "../ADBench/adbench/datasets"
    if adbench.exists():
        registry = build_registry(adbench.resolve())
        cv_nlp = {s.name for s in registry if s.category in ("cv", "nlp")}
        classical = {s.name for s in registry if s.category == "classical"}
    else:
        cv_nlp = CV_NLP_NAMES
        classical = None

    merged: dict = {"completed": {}, "failed": {}}
    sources: list[tuple[str, Path, dict]] = []

    sc_state = load_state(semi_classical_path)
    if classical is not None:
        sc_jobs = filter_completed(sc_state, setting="semi-supervised", datasets=classical)
    else:
        sc_jobs = {
            k: v
            for k, v in sc_state.get("completed", {}).items()
            if v.get("setting") == "semi-supervised" and v.get("dataset") not in cv_nlp
        }
    merged["completed"].update(sc_jobs)
    sources.append(("semi_classical", semi_classical_path, sc_jobs))

    if args.semi_cvnlp_source == "backup":
        cv_state = sc_state
        cv_src_path = semi_classical_path
        cv_label = "semi_cvnlp_from_backup"
    else:
        cv_state = load_state(semi_cvnlp_path)
        cv_src_path = semi_cvnlp_path
        cv_label = "semi_cvnlp"
    cv_jobs = filter_completed(cv_state, setting="semi-supervised", datasets=cv_nlp)
    merged["completed"].update(cv_jobs)
    sources.append((cv_label, cv_src_path, cv_jobs))

    u_state = load_state(unsup_path)
    u_jobs = filter_completed(u_state, setting="unsupervised")
    merged["completed"].update(u_jobs)
    sources.append(("unsup", unsup_path, u_jobs))

    if patch_path and patch_path.exists():
        p_state = load_state(patch_path)
        p_jobs = p_state.get("completed", {})
        merged["completed"].update(p_jobs)
        sources.append(("patch", patch_path, p_jobs))
        print(f"Applied {len(p_jobs)} patch jobs from {patch_path}")

    # Detect key collisions
    total_in = sum(len(j) for _, _, j in sources)
    if len(merged["completed"]) != total_in:
        print(f"NOTE: patch/overlap merge ({total_in} source rows -> {len(merged['completed'])} unique keys)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    print(f"Merged {len(merged['completed'])} jobs -> {out_path}")
    print(f"  semi classical: {len(sc_jobs)}")
    print(f"  semi cv/nlp:    {len(cv_jobs)}")
    print(f"  unsupervised:   {len(u_jobs)}")

    if args.copy_metrics:
        metrics_out = out_path.parent
        copied = 0
        for label, src_completed, jobs in sources:
            src_metrics = src_completed.parent
            copied += copy_metrics(src_metrics, metrics_out, set(jobs.keys()))
        print(f"Copied {copied} per-job metric JSON files to {metrics_out}")


if __name__ == "__main__":
    main()
