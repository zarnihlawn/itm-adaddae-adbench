#!/usr/bin/env python3
"""Validate AdaDDAE v3/v4 Table-1 gates vs DDAE paper and backup."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PUBLISHED = {
    "unsupervised": {"PR-AUC": 32.77, "ROC-AUC": 74.08},
    "semi-supervised": {"PR-AUC": 61.36, "ROC-AUC": 83.17},
}


def load_completed(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("completed", data)


def macro_mean_metrics(completed: dict, setting: str) -> dict:
    """Table-1 macro mean: average per-dataset means over seeds."""
    rows = []
    for job in completed.values():
        if job.get("setting") != setting:
            continue
        m = job.get("metrics_mean", job.get("metrics", {}))
        rows.append({
            "dataset": job["dataset"],
            "PR": m.get("PR-AUC", 0.0) * 100,
            "ROC": m.get("ROC-AUC", 0.0) * 100,
        })
    if not rows:
        return {"PR-AUC": 0.0, "ROC-AUC": 0.0, "n_jobs": 0, "n_datasets": 0}
    df = pd.DataFrame(rows).groupby("dataset").mean()
    return {
        "PR-AUC": float(df["PR"].mean()),
        "ROC-AUC": float(df["ROC"].mean()),
        "n_jobs": len(rows),
        "n_datasets": int(len(df)),
    }


def per_dataset_loss_vs_ref(hybrid: dict, ref: dict, setting: str, threshold: float = 0.5) -> tuple[int, list[str]]:
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


def hard_tail_macro_mean(completed: dict, setting: str, ref_completed: dict, threshold: float = 50.0) -> dict:
    """G6 hard-tail: macro PR on datasets where v4.1 ref PR < threshold."""
    def agg(completed_in, setting_in):
        rows = []
        for j in completed_in.values():
            if j.get("setting") != setting_in:
                continue
            rows.append({"dataset": j["dataset"], "PR": j["metrics_mean"]["PR-AUC"] * 100})
        if not rows:
            return pd.Series(dtype=float)
        return pd.DataFrame(rows).groupby("dataset")["PR"].mean()

    ref_pr = agg(ref_completed, setting)
    hard_ds = ref_pr[ref_pr < threshold].index.tolist()
    cur_pr = agg(completed, setting)
    if not hard_ds:
        return {"macro_pr": 0.0, "n_datasets": 0, "datasets": [], "pass": True}
    hard_cur = cur_pr.reindex(hard_ds).dropna()
    macro = float(hard_cur.mean()) if len(hard_cur) else 0.0
    ref_hard = ref_pr.reindex(hard_ds).dropna()
    delta = macro - float(ref_hard.mean()) if len(ref_hard) else 0.0
    return {
        "macro_pr": macro,
        "ref_macro_pr": float(ref_hard.mean()) if len(ref_hard) else 0.0,
        "delta_vs_ref_pp": delta,
        "n_datasets": len(hard_ds),
        "datasets": sorted(hard_ds),
        "pass": delta >= -0.5,
    }


def lodo_holdout_check(
    completed: dict,
    ref_completed: dict,
    holdout: set[str],
    threshold_pp: float = 0.0,
) -> dict:
    """G7 LODO: holdout datasets must not regress vs v4.1 ref."""
    details = []
    wins = 0
    for setting in ["unsupervised", "semi-supervised"]:
        for ds in holdout:
            def ds_pr(comp, s, d):
                prs = [
                    j["metrics_mean"]["PR-AUC"] * 100
                    for j in comp.values()
                    if j.get("setting") == s and j.get("dataset") == d
                ]
                return float(np.mean(prs)) if prs else None

            cur = ds_pr(completed, setting, ds)
            ref = ds_pr(ref_completed, setting, ds)
            if cur is None or ref is None:
                continue
            delta = cur - ref
            if delta >= threshold_pp:
                wins += 1
            details.append({
                "dataset": ds,
                "setting": setting,
                "cur_pr": cur,
                "ref_pr": ref,
                "delta_pp": delta,
                "pass": delta >= threshold_pp,
            })
    n = len(details)
    return {
        "n_pairs": n,
        "n_pass": sum(1 for d in details if d["pass"]),
        "n_wins": wins,
        "pass": wins >= max(1, int(0.6 * n)) if n else True,
        "holdout": sorted(holdout),
        "details": details,
    }


def per_dataset_backup_loss(hybrid: dict, backup: dict, setting: str, threshold: float = 2.0) -> int:
    n, _ = per_dataset_loss_vs_ref(hybrid, backup, setting, threshold)
    return n


def check_g7_artifact_freshness(completed_path: Path, compare_path: Path, tol: float = 0.05) -> dict:
    """G7: thesis compare_to_ddae.json matches completed.json macro PR."""
    if not compare_path.exists():
        return {"pass": False, "reason": "compare_to_ddae.json missing"}
    completed = load_completed(completed_path)
    compare = json.loads(compare_path.read_text(encoding="utf-8"))
    adadae_rows = {r["setting"]: r for r in compare.get("adadae", [])}
    mismatches = []
    for setting in ["unsupervised", "semi-supervised"]:
        live = macro_mean_metrics(completed, setting)
        cached = adadae_rows.get(setting, {})
        cached_pr = float(cached.get("AdaDDAE_PR_AUC", -1))
        if abs(live["PR-AUC"] - cached_pr) > tol:
            mismatches.append({
                "setting": setting,
                "live_PR": live["PR-AUC"],
                "cached_PR": cached_pr,
                "delta": live["PR-AUC"] - cached_pr,
            })
    return {
        "pass": len(mismatches) == 0,
        "mismatches": mismatches,
        "compare_path": str(compare_path),
    }


def check_g9_merge_audit(audit_path: Path) -> dict:
    """G9: no patch accepted with delta < -0.5 pp."""
    if not audit_path.exists():
        return {"pass": True, "reason": "merge_audit.json missing (skipped)"}
    data = json.loads(audit_path.read_text(encoding="utf-8"))
    bad = data.get("bad_accepts", data.get("n_bad_accepts_lt_0_5pp", 0))
    if isinstance(bad, list):
        n_bad = len(bad)
        bad_list = bad
    else:
        n_bad = int(bad)
        bad_list = data.get("bad_accepts", [])
    return {
        "pass": n_bad == 0,
        "n_bad_accepts_lt_0_5pp": n_bad,
        "bad_accepts": bad_list,
        "audit_path": str(audit_path),
    }


def check_gi1_complete_570(completed: dict, n_seeds: int = 5) -> dict:
    """G-I1: exactly 57 datasets × 2 settings × 5 seeds."""
    keys = list(completed.keys())
    n = len(keys)
    by_ds_setting: dict[tuple[str, str], set[int]] = {}
    for job in completed.values():
        ds = job.get("dataset")
        setting = job.get("setting")
        seed = job.get("seed")
        if ds is None or setting is None or seed is None:
            continue
        by_ds_setting.setdefault((ds, setting), set()).add(int(seed))
    n_pairs = len(by_ds_setting)
    incomplete = [
        f"{ds}/{setting}: seeds={sorted(seeds)}"
        for (ds, setting), seeds in sorted(by_ds_setting.items())
        if len(seeds) != n_seeds
    ]
    ok = n == 570 and n_pairs == 114 and not incomplete
    return {
        "pass": ok,
        "n_jobs": n,
        "n_dataset_setting_pairs": n_pairs,
        "expected_jobs": 570,
        "incomplete_pairs": incomplete[:20],
    }


def _is_allowed_resolved_policy(pol) -> bool:
    """Allow null/static and paradigm_* labels; reject dataset routing ids."""
    if pol in (None, "", "static"):
        return True
    if isinstance(pol, str) and pol.startswith("paradigm_"):
        return True
    return False


def check_gi2_no_routing(completed: dict) -> dict:
    """G-I2: no per-dataset routing; paradigm_* setting labels are allowed."""
    routed = []
    for key, job in completed.items():
        pol = job.get("resolved_policy")
        if not _is_allowed_resolved_policy(pol):
            routed.append({"key": key, "resolved_policy": pol})
    return {"pass": len(routed) == 0, "n_routed": len(routed), "examples": routed[:10]}


def check_gi3_val_only_stop(completed: dict, logs_dir: Path | None = None) -> dict:
    """G-I3: early_stop_metric is val_loss (or train_loss fallback), never test PR."""
    bad = []
    for key, job in completed.items():
        esm = job.get("early_stop_metric")
        if esm is None:
            # older jobs may omit; fail closed for integrity mode
            bad.append({"key": key, "reason": "missing early_stop_metric"})
        elif esm not in ("val_loss", "train_loss"):
            bad.append({"key": key, "reason": f"early_stop_metric={esm!r}"})
    log_bad = []
    if logs_dir and logs_dir.exists():
        for path in logs_dir.glob("*.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("event") == "early_stop":
                    esm = ev.get("early_stop_metric")
                    if esm not in ("val_loss", "train_loss", None):
                        log_bad.append({"file": path.name, "early_stop_metric": esm})
                    if "best_test" in str(ev).lower():
                        log_bad.append({"file": path.name, "reason": "test in early_stop"})
    return {
        "pass": len(bad) == 0 and len(log_bad) == 0,
        "n_bad_jobs": len(bad),
        "examples": bad[:10],
        "n_bad_log_events": len(log_bad),
        "log_examples": log_bad[:10],
    }


def check_gi5_not_guarded_merge(completed_path: Path) -> dict:
    """G-I5: primary completed.json must not be a guarded-merge hybrid artifact."""
    text = completed_path.read_text(encoding="utf-8")
    markers = ("merge_audit", "guarded_merge", "accepted_patches", "delta_pr_pp")
    hits = [m for m in markers if m in text]
    parent = completed_path.parent.parent
    audit = parent / "thesis" / "merge_audit.json"
    # Path name heuristics
    path_l = str(completed_path).lower()
    path_hit = any(x in path_l for x in ("hybrid", "guarded", "v51_hybrid", "v5_hybrid"))
    return {
        "pass": len(hits) == 0 and not path_hit and not audit.exists(),
        "content_markers": hits,
        "path_looks_hybrid": path_hit,
        "merge_audit_present": audit.exists(),
        "path": str(completed_path),
    }


def check_paper_both(compare_path: Path) -> dict:
    """Ship gate: beat published DDAE PR and ROC on both settings."""
    if not compare_path.exists():
        return {"pass": False, "reason": f"missing {compare_path}"}
    data = json.loads(compare_path.read_text(encoding="utf-8"))
    rows = {r["setting"]: r for r in data.get("adadae", [])}
    details = {}
    ok = True
    for setting, paper in PUBLISHED.items():
        row = rows.get(setting)
        if row is None:
            details[setting] = {"pass": False, "reason": "missing setting row"}
            ok = False
            continue
        pr = float(row["AdaDDAE_PR_AUC"])
        roc = float(row["AdaDDAE_ROC_AUC"])
        pr_ok = pr > float(paper["PR-AUC"])
        roc_ok = roc > float(paper["ROC-AUC"])
        details[setting] = {
            "pass": pr_ok and roc_ok,
            "AdaDDAE_PR_AUC": pr,
            "AdaDDAE_ROC_AUC": roc,
            "paper_PR_AUC": paper["PR-AUC"],
            "paper_ROC_AUC": paper["ROC-AUC"],
            "delta_PR": pr - paper["PR-AUC"],
            "delta_ROC": roc - paper["ROC-AUC"],
        }
        if not (pr_ok and roc_ok):
            ok = False
    return {"pass": ok, "settings": details}


def check_ap_pr_consistency(
    completed: dict,
    max_gap_pp: float = 6.0,
    quarantine_bisect: bool = True,
) -> dict:
    """Flag jobs where |AP − PR-AUC| exceeds ``max_gap_pp`` (percentage points).

    Historical v5.1 hybrid Wilt rows rewrote PR-AUC via v31 bisect while AP stayed
    honest (~15 vs ~55). Ship runs must keep AP≈PR (sklearn AP vs trapezoid PR-AUC
    can differ slightly; 5pp is the quarantine threshold).
    """
    bad: list[dict] = []
    bisect_keys: list[str] = []
    n_checked = 0
    for key, job in completed.items():
        if quarantine_bisect and (
            job.get("v31_source")
            or job.get("v31_bisect_candidate")
            or job.get("v31_bisect_seed_pr") is not None
        ):
            bisect_keys.append(key)
        m = job.get("metrics_mean") or job.get("metrics") or {}
        ap = m.get("AP")
        pr = m.get("PR-AUC")
        if ap is None or pr is None:
            continue
        ap_pp = float(ap) * 100.0 if float(ap) <= 1.5 else float(ap)
        pr_pp = float(pr) * 100.0 if float(pr) <= 1.5 else float(pr)
        n_checked += 1
        gap = abs(pr_pp - ap_pp)
        if gap > max_gap_pp:
            bad.append(
                {
                    "key": key,
                    "AP": ap_pp,
                    "PR-AUC": pr_pp,
                    "gap_pp": gap,
                    "v31_source": job.get("v31_source"),
                    "v31_bisect_candidate": job.get("v31_bisect_candidate"),
                }
            )
    # Fail if any large AP/PR gap OR any bisect-marker jobs (quarantine inflated hybrids)
    ok = len(bad) == 0 and (not quarantine_bisect or len(bisect_keys) == 0)
    return {
        "pass": ok,
        "max_gap_pp": max_gap_pp,
        "n_checked": n_checked,
        "n_bad": len(bad),
        "examples": bad[:20],
        "n_bisect_markers": len(bisect_keys),
        "bisect_examples": bisect_keys[:10],
        "note": "Quarantine |AP-PR|>6pp and v31_bisect_* fields; do not trust inflated hybrid PR. "
                "6pp allows small sklearn-AP vs trapezoid-PR differences.",
    }


def run_integrity_gates(
    completed_path: Path,
    compare_path: Path | None = None,
    logs_dir: Path | None = None,
    require_paper_both: bool = False,
) -> dict:
    """G-I1..G-I5 integrity suite for primary / champion recipes."""
    completed = load_completed(completed_path)
    gates = {
        "G-I1_complete_570": check_gi1_complete_570(completed),
        "G-I2_no_routing": check_gi2_no_routing(completed),
        "G-I3_val_only_early_stop": check_gi3_val_only_stop(completed, logs_dir),
        "G-I5_not_guarded_merge": check_gi5_not_guarded_merge(completed_path),
    }
    resolved_compare = compare_path
    if resolved_compare is None:
        default_compare = completed_path.parent.parent / "thesis" / "compare_to_ddae.json"
        if default_compare.exists():
            resolved_compare = default_compare
    if resolved_compare is not None:
        gates["G-I4_artifact_freshness"] = check_g7_artifact_freshness(
            completed_path, resolved_compare
        )
        if require_paper_both:
            gates["G_paper_both"] = check_paper_both(resolved_compare)
    else:
        gates["G-I4_artifact_freshness"] = {
            "pass": False,
            "reason": "compare_to_ddae.json missing (run compare_to_ddae.py after 570)",
        }
        if require_paper_both:
            gates["G_paper_both"] = {"pass": False, "reason": "compare_to_ddae.json missing"}
    gates["all_pass"] = all(g.get("pass") for g in gates.values())
    gates["n_jobs"] = len(completed)
    return gates


def main():
    parser = argparse.ArgumentParser(description="Validate v3/v4 gates")
    parser.add_argument("--completed", required=True)
    parser.add_argument(
        "--backup",
        default="backup/ddae_baseline_570/metrics/completed.json",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--compare",
        default=None,
        help="compare_to_ddae.json for G7 freshness check",
    )
    parser.add_argument(
        "--v31-ref",
        default="results/adadae_v31_hybrid/metrics/completed.json",
        help="Reference hybrid for G6 no-regression vs v3.1",
    )
    parser.add_argument(
        "--v5",
        action="store_true",
        help="Enable v5 gates: G5 beat v4.1, G6 hard-tail, G8 LODO holdout",
    )
    parser.add_argument(
        "--v41-ref",
        default="results/adadae_v41_hybrid/metrics/completed.json",
        help="v4.1 baseline for v5 gates",
    )
    parser.add_argument(
        "--hard-tail-threshold",
        type=float,
        default=50.0,
        help="PR threshold (%%) defining hard-tail datasets",
    )
    parser.add_argument(
        "--v5-strict",
        action="store_true",
        help="G5 requires strictly beating v4.1 (delta > 0 on combined or both settings)",
    )
    parser.add_argument(
        "--merge-audit",
        default=None,
        help="merge_audit.json for G9 check",
    )
    parser.add_argument(
        "--integrity",
        action="store_true",
        help="Run Phase-1 integrity gates G-I1..G-I5 (primary / champion)",
    )
    parser.add_argument(
        "--paper-both",
        action="store_true",
        help="Also require beating published DDAE PR+ROC on both settings",
    )
    parser.add_argument(
        "--logs-dir",
        default=None,
        help="Optional logs dir for G-I3 early-stop scan",
    )
    args = parser.parse_args()

    completed_path = Path(args.completed)
    if not completed_path.is_absolute():
        completed_path = PROJECT_ROOT / completed_path

    if args.integrity:
        compare_path = Path(args.compare) if args.compare else None
        if compare_path is not None and not compare_path.is_absolute():
            compare_path = PROJECT_ROOT / compare_path
        logs_dir = Path(args.logs_dir) if args.logs_dir else completed_path.parent.parent / "logs"
        if not logs_dir.is_absolute():
            logs_dir = PROJECT_ROOT / logs_dir
        gates = run_integrity_gates(
            completed_path,
            compare_path=compare_path,
            logs_dir=logs_dir,
            require_paper_both=bool(args.paper_both),
        )
        print("=== Integrity gates G-I1..G-I5 ===")
        for name, g in gates.items():
            if name in ("all_pass", "n_jobs"):
                continue
            status = "PASS" if g.get("pass") else "FAIL"
            extra = g.get("reason") or g.get("n_jobs") or g.get("n_routed") or ""
            print(f"{name}: {status} {extra}")
        print(f"n_jobs={gates['n_jobs']} ALL PASS: {gates['all_pass']}")
        if args.out:
            out_path = Path(args.out)
            if not out_path.is_absolute():
                out_path = PROJECT_ROOT / out_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(gates, indent=2), encoding="utf-8")
            print(f"Wrote {out_path}")
        sys.exit(0 if gates["all_pass"] else 1)

    backup_path = Path(args.backup)
    if not backup_path.is_absolute():
        backup_path = PROJECT_ROOT / backup_path

    hybrid = load_completed(completed_path)
    backup = load_completed(backup_path)

    gates = {}
    all_pass = True
    for setting in ["unsupervised", "semi-supervised"]:
        m = macro_mean_metrics(hybrid, setting)
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
            "n_datasets": m["n_datasets"],
            "aggregation": "macro_mean",
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

    compare_path = Path(args.compare) if args.compare else completed_path.parent.parent / "thesis" / "compare_to_ddae.json"
    if not compare_path.is_absolute():
        compare_path = PROJECT_ROOT / compare_path
    g7 = check_g7_artifact_freshness(completed_path, compare_path)
    gates["G7_artifact_freshness"] = g7
    all_pass = all_pass and g7["pass"]

    if args.v5:
        v41_path = Path(args.v41_ref)
        if not v41_path.is_absolute():
            v41_path = PROJECT_ROOT / v41_path
        if v41_path.exists():
            v41 = load_completed(v41_path)
            v41_unsup = macro_mean_metrics(v41, "unsupervised")["PR-AUC"]
            v41_semi = macro_mean_metrics(v41, "semi-supervised")["PR-AUC"]
            cur_unsup = gates["unsupervised"]["PR-AUC"]
            cur_semi = gates["semi-supervised"]["PR-AUC"]
            combined_cur = (cur_unsup + cur_semi) / 2.0
            combined_v41 = (v41_unsup + v41_semi) / 2.0
            if args.v5_strict:
                g5_pass = combined_cur > combined_v41 + 1e-6
            else:
                g5_pass = cur_unsup >= v41_unsup - 1e-6 and cur_semi >= v41_semi - 1e-6
            gates["G5_beat_v41"] = {
                "unsup_cur": cur_unsup,
                "unsup_v41": v41_unsup,
                "semi_cur": cur_semi,
                "semi_v41": v41_semi,
                "combined_cur": combined_cur,
                "combined_v41": combined_v41,
                "delta_unsup_pp": cur_unsup - v41_unsup,
                "delta_semi_pp": cur_semi - v41_semi,
                "delta_combined_pp": combined_cur - combined_v41,
                "strict": bool(args.v5_strict),
                "pass": g5_pass,
                "ref": str(v41_path),
            }
            all_pass = all_pass and g5_pass

            ht_detail = {}
            ht_pass = True
            for setting in ["unsupervised", "semi-supervised"]:
                ht = hard_tail_macro_mean(hybrid, setting, v41, threshold=args.hard_tail_threshold)
                ht_detail[setting] = ht
                if not ht["pass"]:
                    ht_pass = False
            gates["G6_hard_tail_macro"] = {**ht_detail, "pass": ht_pass, "threshold_pct": args.hard_tail_threshold}
            all_pass = all_pass and ht_pass

            holdout = {"speech", "Agnews", "Wilt", "celeba", "cardio"}
            g8 = lodo_holdout_check(hybrid, v41, holdout)
            gates["G8_lodo_holdout"] = g8
            all_pass = all_pass and g8["pass"]

    audit_path = Path(args.merge_audit) if args.merge_audit else completed_path.parent.parent / "thesis" / "merge_audit.json"
    if not audit_path.is_absolute():
        audit_path = PROJECT_ROOT / audit_path
    if args.v5 and audit_path.exists():
        g9 = check_g9_merge_audit(audit_path)
        gates["G9_merge_audit"] = g9
        all_pass = all_pass and g9["pass"]

    gates["all_pass"] = all_pass
    gates["n_jobs"] = len(hybrid)
    unsup_m = gates["unsupervised"]["PR-AUC"]
    semi_m = gates["semi-supervised"]["PR-AUC"]
    gates["combined_macro_PR"] = (unsup_m + semi_m) / 2.0

    print("=== Validation gates (macro mean) ===")
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
    print(f"G7 artifact freshness: {'PASS' if g7['pass'] else 'FAIL'}")
    if not g7["pass"] and g7.get("mismatches"):
        for mm in g7["mismatches"]:
            print(f"  stale {mm['setting']}: live {mm['live_PR']:.2f}% vs cached {mm['cached_PR']:.2f}%")
    if args.v5 and "G5_beat_v41" in gates:
        g5 = gates["G5_beat_v41"]
        print(
            f"G5 beat v4.1: unsup {g5['delta_unsup_pp']:+.2f}pp semi {g5['delta_semi_pp']:+.2f}pp "
            f"{'PASS' if g5['pass'] else 'FAIL'}"
        )
        if "G6_hard_tail_macro" in gates:
            ht = gates["G6_hard_tail_macro"]
            print(f"G6 hard-tail macro: {'PASS' if ht['pass'] else 'FAIL'}")
            for setting in ["unsupervised", "semi-supervised"]:
                if setting in ht:
                    h = ht[setting]
                    print(f"  {setting}: {h['macro_pr']:.2f}% (delta {h['delta_vs_ref_pp']:+.2f}pp, n={h['n_datasets']})")
        if "G8_lodo_holdout" in gates:
            g8 = gates["G8_lodo_holdout"]
            print(f"G8 LODO holdout: {g8['n_pass']}/{g8['n_pairs']} pass {'PASS' if g8['pass'] else 'FAIL'}")
        if "G9_merge_audit" in gates:
            g9 = gates["G9_merge_audit"]
            print(f"G9 merge audit: bad_accepts={g9.get('n_bad_accepts_lt_0_5pp', 0)} {'PASS' if g9['pass'] else 'FAIL'}")
    print(f"Combined macro PR: {gates['combined_macro_PR']:.2f}%")
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
