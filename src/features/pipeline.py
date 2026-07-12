"""Feature Tuning Pipeline (FTP) — leak-safe tabular preprocessing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler


@dataclass
class FeaturePolicy:
    scaler: str  # standard | robust
    use_pca: bool
    pca_max_components: int
    pca_variance: float
    clip_outliers: bool
    clip_sigma: float


def infer_policy(
    n_samples: int,
    n_features: int,
    scaler: str = "auto",
    pca_dim_threshold: int = 128,
    pca_max_components: int = 128,
    pca_variance: float = 0.95,
    clip_outliers: bool = True,
    clip_sigma: float = 5.0,
) -> FeaturePolicy:
    if scaler == "auto":
        # Unsupervised / high-d: StandardScaler (matches DDAE paper).
        # Semi-supervised low-d tabular: RobustScaler helps heavy tails.
        chosen = "standard"
    else:
        chosen = scaler
    use_pca = n_features > pca_dim_threshold
    return FeaturePolicy(
        scaler=chosen,
        use_pca=use_pca,
        pca_max_components=min(pca_max_components, n_features, max(2, n_samples - 1)),
        pca_variance=pca_variance,
        clip_outliers=clip_outliers and n_features <= 64,
        clip_sigma=clip_sigma,
    )


class FeatureTuningPipeline:
    """Fit on train only (semi-supervised normals); transform train/test."""

    def __init__(self, policy: FeaturePolicy):
        self.policy = policy
        self.scaler = None
        self.pca = None
        self.clip_bounds_: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.n_features_out_: Optional[int] = None

    def fit(self, X_train: np.ndarray) -> "FeatureTuningPipeline":
        X = np.asarray(X_train, dtype=np.float32)
        if self.policy.clip_outliers:
            mu = X.mean(axis=0)
            std = X.std(axis=0) + 1e-8
            lo = mu - self.policy.clip_sigma * std
            hi = mu + self.policy.clip_sigma * std
            self.clip_bounds_ = (lo.astype(np.float32), hi.astype(np.float32))
            X = np.clip(X, lo, hi)

        if self.policy.scaler == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X).astype(np.float32)

        if self.policy.use_pca:
            n_comp = min(self.policy.pca_max_components, Xs.shape[0] - 1, Xs.shape[1])
            n_comp = max(2, n_comp)
            self.pca = PCA(n_components=min(n_comp, Xs.shape[1]), svd_solver="randomized")
            # Fit full then optionally truncate by variance via explained variance
            self.pca.fit(Xs)
            if self.policy.pca_variance < 1.0:
                cum = np.cumsum(self.pca.explained_variance_ratio_)
                k = int(np.searchsorted(cum, self.policy.pca_variance) + 1)
                k = max(2, min(k, self.pca.n_components_))
                self.pca = PCA(n_components=k, svd_solver="randomized")
                self.pca.fit(Xs)
            Xs = self.pca.transform(Xs).astype(np.float32)

        self.n_features_out_ = Xs.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if self.clip_bounds_ is not None:
            lo, hi = self.clip_bounds_
            X = np.clip(X, lo, hi)
        Xs = self.scaler.transform(X).astype(np.float32)
        if self.pca is not None:
            Xs = self.pca.transform(Xs).astype(np.float32)
        return Xs

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        self.fit(X_train)
        return self.transform(X_train)

    def summary(self) -> Dict[str, Any]:
        return {
            "scaler": self.policy.scaler,
            "use_pca": self.policy.use_pca,
            "n_features_out": self.n_features_out_,
            "pca_components": None if self.pca is None else int(self.pca.n_components_),
            "clip_outliers": self.policy.clip_outliers,
        }
