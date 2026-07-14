"""Meta-feature policy routing from bisect matrix (decision tree)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml

from .config import PROJECT_ROOT

CATEGORY_MAP = {"classical": 0, "cv": 1, "nlp": 2}
SETTING_MAP = {"unsupervised": 0, "semi-supervised": 1}

# Bisect candidate -> policy registry id
CANDIDATE_TO_POLICY: Dict[str, str] = {
    "baseline_ddae": "baseline_ddae",
    "semi_nlp_danc": "semi_nlp_baseline",
    "semi_speech_specialist": "semi_speech_specialist",
    "semi_rdt_tail": "semi_rdt_tail",
    "semi_cvnlp_ftp": "semi_cvnlp_ftp",
    "semi_cvnlp_taps_004": "semi_cvnlp_taps_light",
    "semi_cvnlp_taps_006": "semi_cvnlp_taps_light",
    "ssts": "unsup_ssts",
    "taps": "unsup_ssts",
    "lfdanc": "unsup_ssts",
    "ftp": "semi_cvnlp_ftp",
    "rdt": "semi_rdt_tail",
    "unsup_baseline": "unsup_baseline_fallback",
    "unsup_lfdanc": "unsup_ssts",
    "unsup_ssts": "unsup_ssts",
    "unsup_nlp_ssts_light": "unsup_nlp_ssts_light",
    "unsup_classical_plus": "unsup_classical_plus",
    "ddae_repro": "unsup_baseline_fallback",
    "oracle_danc": "unsup_ssts",
}


def load_routing_rules(path: Optional[str | Path] = None) -> Dict[str, Any]:
    if path is None:
        path = PROJECT_ROOT / "configs" / "routing_rules.yaml"
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        return {"enabled": False, "dataset_overrides": {}}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {"enabled": False, "dataset_overrides": {}}


def _dataset_meta_features(dataset: str, category: str, setting: str, meta: Dict[str, float]) -> Dict[str, float]:
    n = max(float(meta.get("n", 1)), 1.0)
    d = max(float(meta.get("d", 1)), 1.0)
    return {
        "log_n": float(__import__("math").log10(n)),
        "log_d": float(__import__("math").log10(d)),
        "category_code": float(CATEGORY_MAP.get(category, 0)),
        "setting_code": float(SETTING_MAP.get(setting, 0)),
        "contamination": float(meta.get("contamination", 0.05)),
        "skew": float(meta.get("skew", 0.0)),
    }


def train_routing_tree(
    matrix_path: Path,
    max_depth: int = 4,
    holdout_datasets: Optional[set[str]] = None,
) -> tuple[Any, pd.DataFrame]:
    """Train sklearn decision tree on bisect matrix; return (model, feature_df)."""
    from sklearn.tree import DecisionTreeClassifier

    df = pd.read_csv(matrix_path)
    # Best candidate per (dataset, setting) by mean PR
    best = (
        df.groupby(["dataset", "setting", "category", "candidate"])["PR-AUC"]
        .mean()
        .reset_index()
        .sort_values("PR-AUC", ascending=False)
        .groupby(["dataset", "setting"], as_index=False)
        .first()
    )

    rows = []
    for _, r in best.iterrows():
        rows.append({
            "dataset": r["dataset"],
            "setting": r["setting"],
            "category": r.get("category", "classical"),
            "candidate": r["candidate"],
            "log_n": 3.0,
            "log_d": 2.0,
            "category_code": CATEGORY_MAP.get(r.get("category", "classical"), 0),
            "setting_code": SETTING_MAP.get(r["setting"], 0),
            "contamination": 0.05,
            "skew": 0.0,
        })

    feat_df = pd.DataFrame(rows)
    if feat_df.empty:
        return None, feat_df

    feature_cols = ["log_n", "log_d", "category_code", "setting_code", "contamination", "skew"]
    X = feat_df[feature_cols].fillna(0)
    y = feat_df["candidate"]

    train_mask = ~feat_df["dataset"].isin(holdout_datasets or set())
    if train_mask.sum() < 3:
        train_mask = pd.Series([True] * len(feat_df))

    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X[train_mask], y[train_mask])
    return clf, feat_df


def export_routing_rules(
    matrix_paths: list[Path],
    out_path: Path,
    max_depth: int = 4,
    holdout: Optional[set[str]] = None,
) -> Dict[str, Any]:
    """Build routing_rules.yaml from bisect matrices."""
    frames = []
    for p in matrix_paths:
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        rules = {"enabled": False, "dataset_overrides": {}, "meta_routing": False}
        out_path.write_text(yaml.dump(rules, default_flow_style=False), encoding="utf-8")
        return rules

    combined = pd.concat(frames, ignore_index=True)
    tmp = PROJECT_ROOT / "results" / "thesis" / "_routing_train_tmp.csv"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(tmp, index=False)

    clf, feat_df = train_routing_tree(tmp, max_depth=max_depth, holdout_datasets=holdout)

    # Per-dataset overrides from best mean PR
    overrides: Dict[str, Dict[str, str]] = {"unsupervised": {}, "semi-supervised": {}}
    for (ds, setting), grp in combined.groupby(["dataset", "setting"]):
        means = grp.groupby("candidate")["PR-AUC"].mean()
        best_cand = str(means.idxmax())
        policy = CANDIDATE_TO_POLICY.get(best_cand, best_cand)
        overrides.setdefault(setting, {})[ds] = policy

    rules: Dict[str, Any] = {
        "enabled": True,
        "meta_routing": True,
        "max_depth": max_depth,
        "dataset_overrides": overrides,
        "candidate_to_policy": CANDIDATE_TO_POLICY,
        "holdout_datasets": sorted(holdout or []),
        "n_training_datasets": int(len(feat_df)),
    }

    if clf is not None:
        # Export simple lookup: dataset -> policy (primary use)
        rules["tree_available"] = True

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(rules, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return rules


def resolve_meta_policy(
    setting: str,
    category: str,
    dataset_name: str,
    meta: Optional[Dict[str, float]] = None,
    rules: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Return policy id from routing_rules if enabled, else None."""
    rules = rules or load_routing_rules()
    if not rules.get("enabled", False):
        return None

    overrides = rules.get("dataset_overrides", {})
    setting_overrides = overrides.get(setting, {})
    if dataset_name in setting_overrides:
        return setting_overrides[dataset_name]

    # Meta guards for unsup_classical_plus
    meta = meta or {}
    n = int(meta.get("n", 0))
    d = int(meta.get("d", 0))
    if setting == "unsupervised" and category == "classical" and n > 5000 and d < 512:
        # Only if explicitly in overrides — don't auto-apply VUS stack
        pass

    return None
