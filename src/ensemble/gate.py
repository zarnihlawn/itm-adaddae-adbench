"""GATE: Train-only ensemble gating over AdaDDAE / DDAE / IsolationForest / kNN-DTE."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@dataclass
class GateDecision:
    winner: str
    weights: Dict[str, float]
    disagreement: float
    fallback: bool


def _rank_normalize(scores: np.ndarray) -> np.ndarray:
    """Convert scores to [0,1] rank scale (higher = more anomalous)."""
    n = len(scores)
    if n < 2:
        return np.zeros(n, dtype=np.float64)
    order = np.argsort(scores)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, n)
    return ranks


def fit_isolation_forest(X_train: np.ndarray, seed: int = 42) -> Tuple[IsolationForest, StandardScaler]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    clf = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(Xs)
    return clf, scaler


def isolation_scores(clf: IsolationForest, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    Xs = scaler.transform(X)
    raw = -clf.decision_function(Xs)
    return raw.astype(np.float64)


def knn_dte_proxy_scores(X_train: np.ndarray, X: np.ndarray, k: int = 5) -> np.ndarray:
    """Lightweight kNN distance to training normals (train-only memory)."""
    from sklearn.neighbors import NearestNeighbors

    n_train = min(4096, X_train.shape[0])
    rng = np.random.RandomState(0)
    if X_train.shape[0] > n_train:
        idx = rng.choice(X_train.shape[0], n_train, replace=False)
        mem = X_train[idx]
    else:
        mem = X_train
    nn = NearestNeighbors(n_neighbors=min(k, mem.shape[0]), metric="euclidean")
    nn.fit(mem)
    dist, _ = nn.kneighbors(X)
    return dist.mean(axis=1).astype(np.float64)


def score_consistency_on_normals(
    score_dict: Dict[str, np.ndarray],
) -> Tuple[Dict[str, float], float]:
    """
    Per-model weight from inverse rank-variance on training normals.
    Returns (weights, disagreement).
    """
    if not score_dict:
        return {}, 1.0
    ranked = {name: _rank_normalize(s) for name, s in score_dict.items()}
    vars_ = {name: float(np.var(r) + 1e-8) for name, r in ranked.items()}
    inv = {name: 1.0 / v for name, v in vars_.items()}
    total = sum(inv.values())
    weights = {name: v / total for name, v in inv.items()}
    mats = np.stack(list(ranked.values()), axis=0)
    disagreement = float(np.std(mats, axis=0).mean())
    return weights, disagreement


def pick_winner_model(
    train_normal_scores: Dict[str, np.ndarray],
) -> Tuple[str, float]:
    """Winner-take-all: lowest rank-variance model on train normals."""
    if not train_normal_scores:
        return "adadae", 1.0
    vars_: Dict[str, float] = {}
    for name, s in train_normal_scores.items():
        r = _rank_normalize(np.asarray(s, dtype=np.float64))
        vars_[name] = float(np.var(r) + 1e-8)
    winner = min(vars_, key=vars_.get)
    mats = np.stack([_rank_normalize(np.asarray(s, dtype=np.float64)) for s in train_normal_scores.values()])
    disagreement = float(np.std(mats, axis=0).mean())
    return winner, disagreement


def gate_winner_predict(
    score_dict: Dict[str, np.ndarray],
    train_normal_scores: Dict[str, np.ndarray],
    disagreement_threshold: float = 0.15,
    conformal_fallback: str = "ddae",
) -> Tuple[np.ndarray, GateDecision]:
    """Winner-take-all GATE: use single best model on test (no rank blend)."""
    winner, disagreement = pick_winner_model(train_normal_scores)
    if disagreement > disagreement_threshold:
        fallback_scores = score_dict.get(conformal_fallback, score_dict.get("adadae"))
        return fallback_scores, GateDecision(
            winner=conformal_fallback,
            weights={conformal_fallback: 1.0},
            disagreement=disagreement,
            fallback=True,
        )
    if winner not in score_dict:
        winner = "adadae"
    return score_dict[winner], GateDecision(
        winner=winner,
        weights={winner: 1.0},
        disagreement=disagreement,
        fallback=False,
    )


def gate_ensemble_predict(
    adadae_scores: np.ndarray,
    ddae_scores: np.ndarray,
    if_scores: np.ndarray,
    knn_scores: np.ndarray,
    train_normal_scores: Dict[str, np.ndarray],
    disagreement_threshold: float = 0.15,
    conformal_fallback: str = "ddae",
) -> Tuple[np.ndarray, GateDecision]:
    """
    Fuse test scores using weights learned from train-normal consistency.
    Falls back to DDAE when ensemble disagreement exceeds threshold.
    """
    weights, disagreement = score_consistency_on_normals(train_normal_scores)
    if not weights or disagreement > disagreement_threshold:
        w = {conformal_fallback: 1.0}
        fused = ddae_scores if conformal_fallback == "ddae" else adadae_scores
        return fused, GateDecision(
            winner=conformal_fallback,
            weights=w,
            disagreement=disagreement,
            fallback=True,
        )

    test_ranked = {
        "adadae": _rank_normalize(adadae_scores),
        "ddae": _rank_normalize(ddae_scores),
        "iforest": _rank_normalize(if_scores),
        "knn_dte": _rank_normalize(knn_scores),
    }
    key_map = {"adadae": "adadae", "ddae": "ddae", "iforest": "iforest", "knn_dte": "knn_dte"}
    fused = np.zeros(len(adadae_scores), dtype=np.float64)
    for name, w in weights.items():
        if name in test_ranked:
            fused += w * test_ranked[name]
    winner = max(weights, key=weights.get)
    return fused, GateDecision(
        winner=winner,
        weights=weights,
        disagreement=disagreement,
        fallback=False,
    )


def build_train_normal_scores(
    X_train: np.ndarray,
    adadae_fn,
    ddae_fn,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Collect per-model scores on training normals for GATE calibration."""
    n_cal = min(512, X_train.shape[0])
    rng = np.random.RandomState(seed)
    idx = rng.choice(X_train.shape[0], n_cal, replace=False)
    X_cal = X_train[idx]
    if_clf, if_scaler = fit_isolation_forest(X_train, seed=seed)
    return {
        "adadae": np.asarray(adadae_fn(X_cal), dtype=np.float64),
        "ddae": np.asarray(ddae_fn(X_cal), dtype=np.float64),
        "iforest": isolation_scores(if_clf, if_scaler, X_cal),
        "knn_dte": knn_dte_proxy_scores(X_train, X_cal),
    }
