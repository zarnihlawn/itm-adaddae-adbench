"""OMNI — rich train-only meta features for AdaDDAE-4 regime gates."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


OMNI_KEYS: List[str] = [
    "log_n",
    "log_d",
    "log_n_over_d",
    "contamination",
    "skewness",
    "kurtosis",
    "intrinsic_dim_ratio",
    "sparsity",
    "cluster_sep",
    "pca_energy_8",
    "pca_energy_32",
    "cat_classical",
    "cat_cv",
    "cat_nlp",
    "is_tiny",
    "is_huge",
    "is_highd",
    "is_heavy",
    "is_rare",
    "is_multi",
    "is_sparse",
]


def _kurtosis(X: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    std = X.std(axis=0) + 1e-12
    m4 = ((X - mu) ** 4).mean(axis=0)
    return float(np.mean(m4 / (std**4)))


def _sparsity(X: np.ndarray) -> float:
    return float(1.0 - (np.count_nonzero(X) / max(X.size, 1)))


def _pca_energy(X: np.ndarray, k: int, max_samples: int = 2000) -> float:
    from sklearn.decomposition import PCA

    n, d = X.shape
    if n < 3 or d < 2:
        return 1.0
    rng = np.random.RandomState(0)
    if n > max_samples:
        X = X[rng.choice(n, max_samples, replace=False)]
    kk = min(k, X.shape[0] - 1, X.shape[1])
    if kk < 1:
        return 1.0
    pca = PCA(n_components=kk, svd_solver="randomized", random_state=0)
    pca.fit(X)
    return float(pca.explained_variance_ratio_.sum())


def _cluster_sep(X: np.ndarray, max_n: int = 2000) -> float:
    from sklearn.cluster import MiniBatchKMeans

    n = X.shape[0]
    if n < 8:
        return 0.0
    rng = np.random.RandomState(0)
    idx = rng.choice(n, min(max_n, n), replace=False)
    Xs = X[idx].astype(np.float64)
    Xs = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-8)
    try:
        km = MiniBatchKMeans(n_clusters=2, random_state=0, n_init=3, batch_size=256)
        lab = km.fit_predict(Xs)
        c0 = Xs[lab == 0].mean(0)
        c1 = Xs[lab == 1].mean(0)
        sep = float(np.linalg.norm(c0 - c1))
        w = float(((Xs[lab == 0] - c0) ** 2).sum() + ((Xs[lab == 1] - c1) ** 2).sum())
        return sep / (np.sqrt(w / max(len(Xs), 1)) + 1e-8)
    except Exception:
        return 0.0


def enrich_omni_meta(
    meta: Dict[str, float],
    X: Optional[np.ndarray] = None,
    category: str = "classical",
) -> Dict[str, float]:
    """Augment DANC meta with OMNI fields (train-only)."""
    out = dict(meta)
    n = float(out.get("n", 1.0))
    d = float(out.get("d", 1.0))
    contam = float(out.get("contamination", 0.05))
    skew = float(out.get("skewness", 0.0))
    idim = float(out.get("intrinsic_dim", d))

    out["log_n_over_d"] = float(np.log1p(n) - np.log1p(d))
    out["intrinsic_dim_ratio"] = idim / max(d, 1.0)
    out["cat_classical"] = 1.0 if category == "classical" else 0.0
    out["cat_cv"] = 1.0 if category == "cv" else 0.0
    out["cat_nlp"] = 1.0 if category == "nlp" else 0.0

    if X is not None:
        out["kurtosis"] = _kurtosis(X)
        out["sparsity"] = _sparsity(X)
        out["cluster_sep"] = _cluster_sep(X)
        out["pca_energy_8"] = _pca_energy(X, 8)
        out["pca_energy_32"] = _pca_energy(X, 32)
    else:
        out.setdefault("kurtosis", 3.0)
        out.setdefault("sparsity", 0.0)
        out.setdefault("cluster_sep", 0.0)
        out.setdefault("pca_energy_8", 0.5)
        out.setdefault("pca_energy_32", 0.8)

    out["is_tiny"] = 1.0 if n < 300 else 0.0
    out["is_huge"] = 1.0 if n >= 50_000 else 0.0
    out["is_highd"] = 1.0 if d >= 128 or category in ("cv", "nlp") else 0.0
    out["is_heavy"] = 1.0 if contam >= 0.2 else 0.0
    out["is_rare"] = 1.0 if contam <= 0.02 else 0.0
    out["is_multi"] = 1.0 if float(out.get("cluster_sep", 0.0)) > 1.5 else 0.0
    out["is_sparse"] = 1.0 if float(out.get("sparsity", 0.0)) > 0.7 else 0.0
    out["is_skew"] = 1.0 if skew >= 5.0 else 0.0
    return out


def omni_to_phi(meta: Dict[str, float], dim: int = 16) -> np.ndarray:
    """Fixed-length OMNI vector (pad/truncate to dim for CHRONOS/ATLAS)."""
    vals = []
    for k in OMNI_KEYS:
        vals.append(float(meta.get(k, 0.0)))
    arr = np.asarray(vals, dtype=np.float32)
    if arr.size < dim:
        arr = np.pad(arr, (0, dim - arr.size))
    return arr[:dim]


def regime_gate_flags(meta: Dict[str, float]) -> Dict[str, bool]:
    """Train-only meta → which AdaDDAE-4/5 regime modules should activate."""
    high_d = bool(meta.get("is_highd", 0.0) >= 0.5 or float(meta.get("d", 0.0)) >= 64)
    classical_hard = bool(
        float(meta.get("d", 0.0)) >= 32
        and float(meta.get("n", 0.0)) < 5000
        and float(meta.get("contamination", 0.0)) < 0.1
    )
    return {
        "use_nano": bool(meta.get("is_tiny", 0.0) >= 0.5),
        "use_torrent": bool(meta.get("is_huge", 0.0) >= 0.5),
        "use_prism": bool(meta.get("is_highd", 0.0) >= 0.5),
        "use_polis": bool(meta.get("is_multi", 0.0) >= 0.5),
        "use_sieve": bool(meta.get("is_heavy", 0.0) >= 0.5),
        "use_needle": bool(meta.get("is_rare", 0.0) >= 0.5),
        "use_sparse": bool(meta.get("is_sparse", 0.0) >= 0.5),
        "use_robust": bool(meta.get("is_skew", 0.0) >= 0.5 or meta.get("skewness", 0.0) >= 3.0),
        # A5 selective keepers (only when auto_regime_gates=True)
        "use_geode": high_d or classical_hard,
        "use_orbis": high_d,
    }
