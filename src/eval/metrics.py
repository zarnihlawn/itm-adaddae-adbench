"""Anomaly detection metrics matching AnoDDAE utils."""
from __future__ import annotations

from typing import Dict, Union

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, auc


def evaluate_anomaly_detection(
    scores: Union[np.ndarray, list],
    labels: Union[np.ndarray, list],
) -> Dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel()
    if len(np.unique(labels)) < 2:
        return {"AP": float("nan"), "ROC-AUC": float("nan"), "PR-AUC": float("nan")}

    ap = float(average_precision_score(labels, scores))
    roc_auc = float(roc_auc_score(labels, scores))
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = float(auc(recall, precision))
    return {"AP": ap, "ROC-AUC": roc_auc, "PR-AUC": pr_auc}


def mean_std_metrics(rows: list[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = ["AP", "ROC-AUC", "PR-AUC"]
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows if k in r and np.isfinite(r[k])], dtype=np.float64)
        if len(vals) == 0:
            out[k] = {"mean": float("nan"), "std": float("nan")}
        else:
            out[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0))}
    return out
