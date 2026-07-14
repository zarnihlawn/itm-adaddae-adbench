"""DAMP: Dataset-Adaptive Meta-Diffusion Policy (LODO meta-learning)."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT
from .policy_meta import CANDIDATE_TO_POLICY, CATEGORY_MAP, HOLDOUT_DEFAULT, SETTING_MAP, _is_valid_override

DAMP_MODEL_PATH = PROJECT_ROOT / "configs" / "damp_model.pkl"
DAMP_META_PATH = PROJECT_ROOT / "configs" / "damp_meta.json"

FEATURE_COLS = [
    "log_n",
    "log_d",
    "category_code",
    "setting_code",
    "contamination",
    "skew",
    "intrinsic_dim",
    "tail_rate",
    "effective_rank",
    "snr_proxy",
]


def _extended_meta_features(
    dataset: str,
    category: str,
    setting: str,
    meta: Dict[str, float],
) -> Dict[str, float]:
    import math

    n = max(float(meta.get("n", 1000)), 1.0)
    d = max(float(meta.get("d", 10)), 1.0)
    contam = float(meta.get("contamination", meta.get("contamination_est", 0.05)))
    skew = float(meta.get("skewness", meta.get("skew", 0.0)))
    idim = float(meta.get("intrinsic_dim", min(d, n)))
    tail_rate = float(meta.get("tail_rate", contam))
    eff_rank = float(meta.get("effective_rank", idim))
    snr_proxy = float(meta.get("snr_proxy", 1.0 / (1.0 + idim / max(d, 1.0))))
    return {
        "log_n": math.log10(n),
        "log_d": math.log10(d),
        "category_code": float(CATEGORY_MAP.get(category, 0)),
        "setting_code": float(SETTING_MAP.get(setting, 0)),
        "contamination": contam,
        "skew": skew,
        "intrinsic_dim": idim,
        "tail_rate": tail_rate,
        "effective_rank": eff_rank,
        "snr_proxy": snr_proxy,
        "dataset": dataset,
        "category": category,
        "setting": setting,
    }


def _load_bisect_training_frames(matrix_paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in matrix_paths:
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_damp_training_data(
    matrix_paths: Optional[list[Path]] = None,
    dataset_meta: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.DataFrame:
    """Best bisect candidate per (dataset, setting) with extended meta-features."""
    if matrix_paths is None:
        matrix_paths = [
            PROJECT_ROOT / "results" / "thesis" / "v4_unsup_bisect_matrix.csv",
            PROJECT_ROOT / "results" / "thesis" / "v31_semi_tail_matrix.csv",
            PROJECT_ROOT / "results" / "thesis" / "v41_unsup_fallback_bisect_matrix.csv",
        ]
    combined = _load_bisect_training_frames(matrix_paths)
    if combined.empty:
        return pd.DataFrame()

    best = (
        combined.groupby(["dataset", "setting", "category", "candidate"])["PR-AUC"]
        .mean()
        .reset_index()
        .sort_values("PR-AUC", ascending=False)
        .groupby(["dataset", "setting"], as_index=False)
        .first()
    )

    rows = []
    for _, r in best.iterrows():
        ds = str(r["dataset"])
        setting = str(r["setting"])
        category = str(r.get("category", "classical"))
        meta = (dataset_meta or {}).get(f"{ds}:{setting}", {})
        meta.setdefault("n", 1000)
        meta.setdefault("d", 10)
        feats = _extended_meta_features(ds, category, setting, meta)
        feats["candidate"] = str(r["candidate"])
        feats["label_pr"] = float(r["PR-AUC"])
        rows.append(feats)
    return pd.DataFrame(rows)


def train_damp_model(
    feat_df: pd.DataFrame,
    holdout_datasets: Optional[set[str]] = None,
    max_depth: int = 5,
) -> tuple[Any, dict]:
    """Train sklearn classifier on bisect winners; return (model, report)."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import LabelEncoder

    holdout = holdout_datasets or HOLDOUT_DEFAULT
    if feat_df.empty or len(feat_df) < 5:
        return None, {"error": "insufficient training data", "n_samples": len(feat_df)}

    le = LabelEncoder()
    y = le.fit_transform(feat_df["candidate"])
    X = feat_df[FEATURE_COLS].fillna(0.0).values.astype(np.float64)

    train_mask = ~feat_df["dataset"].isin(holdout)
    if train_mask.sum() < 3:
        train_mask = pd.Series([True] * len(feat_df))

    clf = GradientBoostingClassifier(
        max_depth=max_depth,
        n_estimators=80,
        learning_rate=0.1,
        random_state=42,
    )
    clf.fit(X[train_mask], y[train_mask])

    holdout_mask = feat_df["dataset"].isin(holdout)
    holdout_report: dict = {"n_holdout": int(holdout_mask.sum()), "wins": 0, "details": []}
    if holdout_mask.any() and hasattr(clf, "predict"):
        for idx in feat_df.index[holdout_mask]:
            row = feat_df.loc[idx]
            x = row[FEATURE_COLS].fillna(0.0).values.astype(np.float64).reshape(1, -1)
            pred_cand = le.inverse_transform(clf.predict(x))[0]
            true_cand = row["candidate"]
            win = pred_cand == true_cand
            if win:
                holdout_report["wins"] += 1
            holdout_report["details"].append({
                "dataset": row["dataset"],
                "setting": row["setting"],
                "predicted": pred_cand,
                "oracle": true_cand,
                "match": win,
            })

    bundle = {"model": clf, "label_encoder": le, "feature_cols": FEATURE_COLS}
    report = {
        "n_train": int(train_mask.sum()),
        "n_classes": len(le.classes_),
        "classes": le.classes_.tolist(),
        "holdout": holdout_report,
        "holdout_datasets": sorted(holdout),
    }
    return bundle, report


def save_damp_model(bundle: dict, report: dict, model_path: Optional[Path] = None) -> Path:
    model_path = model_path or DAMP_MODEL_PATH
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    meta_path = model_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return model_path


def load_damp_model(model_path: Optional[Path] = None) -> Optional[dict]:
    model_path = model_path or DAMP_MODEL_PATH
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not model_path.exists():
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)


def resolve_damp_policy(
    setting: str,
    category: str,
    dataset_name: str,
    meta: Optional[Dict[str, float]] = None,
    bundle: Optional[dict] = None,
) -> Optional[str]:
    """Predict policy id from DAMP meta-model."""
    bundle = bundle or load_damp_model()
    if bundle is None:
        return None

    meta = meta or {}
    feats = _extended_meta_features(dataset_name, category, setting, meta)
    X = np.array([[feats[c] for c in bundle["feature_cols"]]], dtype=np.float64)
    clf = bundle["model"]
    le = bundle["label_encoder"]
    try:
        cand = le.inverse_transform(clf.predict(X))[0]
    except Exception:
        return None

    policy = CANDIDATE_TO_POLICY.get(str(cand), str(cand))
    if not _is_valid_override(setting, category, policy):
        return None
    return policy


def train_and_export(
    matrix_paths: Optional[list[Path]] = None,
    holdout: Optional[set[str]] = None,
    out_path: Optional[Path] = None,
) -> dict:
    """End-to-end DAMP training from bisect matrices."""
    feat_df = build_damp_training_data(matrix_paths=matrix_paths)
    bundle, report = train_damp_model(feat_df, holdout_datasets=holdout)
    if bundle is None:
        return report
    path = save_damp_model(bundle, report, out_path)
    report["model_path"] = str(path)
    DAMP_META_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
