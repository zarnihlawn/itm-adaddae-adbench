#!/usr/bin/env python3
"""Regression-guarded patch merge for AdaDDAE v5.1."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.merge_completed import load_state


def job_pr(job: dict) -> float:
    m = job.get("metrics_mean", job.get("metrics", {}))
    return float(m.get("PR-AUC", 0.0)) * 100.0


def apply_guarded_patches(
    base: dict[str, Any],
    patch_jobs: dict[str, Any],
    guard_epsilon: float = 0.1,
    layer: str = "patch",
) -> tuple[dict[str, Any], list[dict]]:
    """Return (updated_completed, audit_rows). Apply patch only if PR delta >= -epsilon pp."""
    audit: list[dict] = []
    out = dict(base)
    for key, job in patch_jobs.items():
        base_job = out.get(key)
        patch_pr = job_pr(job)
        if base_job is None:
            out[key] = job
            audit.append({
                "key": key,
                "layer": layer,
                "action": "accept_new",
                "base_pr": None,
                "patch_pr": patch_pr,
                "delta_pp": None,
            })
            continue
        base_pr = job_pr(base_job)
        delta = patch_pr - base_pr
        if delta >= -guard_epsilon:
            out[key] = job
            audit.append({
                "key": key,
                "layer": layer,
                "action": "accept",
                "base_pr": base_pr,
                "patch_pr": patch_pr,
                "delta_pp": delta,
            })
        else:
            audit.append({
                "key": key,
                "layer": layer,
                "action": "reject",
                "base_pr": base_pr,
                "patch_pr": patch_pr,
                "delta_pp": delta,
            })
    return out, audit


def merge_layers_guarded(
    base_path: Path,
    patch_paths: list[tuple[str, Path]],
    guard_epsilon: float = 0.1,
) -> tuple[dict, list[dict]]:
    state = load_state(base_path)
    completed = dict(state.get("completed", {}))
    all_audit: list[dict] = []
    for layer, path in patch_paths:
        if not path.exists():
            continue
        patch_state = load_state(path)
        patch_jobs = patch_state.get("completed", {})
        completed, audit = apply_guarded_patches(
            completed, patch_jobs, guard_epsilon=guard_epsilon, layer=layer
        )
        all_audit.extend(audit)
    return {"completed": completed, "failed": {}}, all_audit


def main():
    parser = argparse.ArgumentParser(description="Guarded v5.1 merge")
    parser.add_argument("--base", default="results/adadae_v41_hybrid/metrics/completed.json")
    parser.add_argument("--phase1", default="results/adadae_v51_phase1/metrics/completed.json")
    parser.add_argument("--mce", default="results/adadae_v51_mce/metrics/completed.json")
    parser.add_argument("--smc", default="results/adadae_v51_smc/metrics/completed.json")
    parser.add_argument("--gate", default="")
    parser.add_argument("--guard-epsilon", type=float, default=0.1)
    parser.add_argument("--out", default="results/adadae_v51_hybrid/metrics/completed.json")
    parser.add_argument("--audit-out", default="results/adadae_v51_hybrid/thesis/merge_audit.json")
    args = parser.parse_args()

    base_path = PROJECT_ROOT / args.base
    patches: list[tuple[str, Path]] = [
        ("phase1", PROJECT_ROOT / args.phase1),
        ("mce", PROJECT_ROOT / args.mce),
        ("smc", PROJECT_ROOT / args.smc),
    ]
    if args.gate:
        patches.append(("gate", PROJECT_ROOT / args.gate))

    merged, audit = merge_layers_guarded(base_path, patches, guard_epsilon=args.guard_epsilon)

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    audit_path = PROJECT_ROOT / args.audit_out
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    accepted = sum(1 for a in audit if a["action"] in ("accept", "accept_new"))
    rejected = sum(1 for a in audit if a["action"] == "reject")
    bad_accepts = [a for a in audit if a["action"] == "accept" and a.get("delta_pp", 0) < -0.5]
    summary = {
        "base": str(base_path),
        "out": str(out_path),
        "guard_epsilon_pp": args.guard_epsilon,
        "n_accepted": accepted,
        "n_rejected": rejected,
        "n_bad_accepts_lt_0_5pp": len(bad_accepts),
        "bad_accepts": bad_accepts,
        "entries": audit,
    }
    audit_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Merged {len(merged['completed'])} jobs -> {out_path}")
    print(f"Guarded merge: {accepted} accepted, {rejected} rejected")
    print(f"Audit -> {audit_path}")


if __name__ == "__main__":
    main()
