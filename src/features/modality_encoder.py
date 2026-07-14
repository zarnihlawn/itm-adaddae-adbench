"""MCE: Modality-Conditional Encoders (train-only, leak-safe)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import RobustScaler, StandardScaler


def target_dim(n_samples: int, n_features: int, modality: str) -> int:
    """Deterministic output dimension from (n, d, modality)."""
    if modality == "nlp":
        base = min(128, max(16, n_features // 4))
    elif modality == "cv":
        base = 128
    else:
        base = min(64, max(8, n_features))
    return max(2, min(base, n_samples - 1, n_features))


@dataclass
class ModalityEncoderPolicy:
    modality: str
    output_dim: int
    use_svd: bool
    use_random_projection: bool


class ModalityEncoder:
    """Fit on training normals only; transform train/test before FTP/DANC."""

    def __init__(self, modality: str = "classical", output_dim: Optional[int] = None):
        self.modality = modality
        self.output_dim = output_dim
        self.scaler: Optional[StandardScaler | RobustScaler] = None
        self.svd: Optional[TruncatedSVD] = None
        self.rp_matrix_: Optional[np.ndarray] = None
        self.n_features_out_: Optional[int] = None
        self.policy_: Optional[ModalityEncoderPolicy] = None

    def _infer_policy(self, n: int, d: int) -> ModalityEncoderPolicy:
        out_d = self.output_dim or target_dim(n, d, self.modality)
        if self.modality == "nlp":
            return ModalityEncoderPolicy(
                modality="nlp",
                output_dim=out_d,
                use_svd=True,
                use_random_projection=False,
            )
        if self.modality == "cv":
            return ModalityEncoderPolicy(
                modality="cv",
                output_dim=out_d,
                use_svd=False,
                use_random_projection=True,
            )
        return ModalityEncoderPolicy(
            modality="classical",
            output_dim=out_d,
            use_svd=d > 256,
            use_random_projection=False,
        )

    def fit(self, X_train: np.ndarray, seed: int = 42) -> "ModalityEncoder":
        X = np.asarray(X_train, dtype=np.float32)
        n, d = X.shape
        self.policy_ = self._infer_policy(n, d)
        pol = self.policy_

        if self.modality == "cv":
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X).astype(np.float32)

        if pol.use_svd and d > pol.output_dim:
            n_comp = min(pol.output_dim, n - 1, d)
            self.svd = TruncatedSVD(n_components=max(2, n_comp), random_state=seed)
            Xs = self.svd.fit_transform(Xs).astype(np.float32)
        elif pol.use_random_projection and d > pol.output_dim:
            rng = np.random.RandomState(seed)
            k = min(pol.output_dim, d)
            self.rp_matrix_ = rng.randn(d, k).astype(np.float32) / np.sqrt(k)
            Xs = (Xs @ self.rp_matrix_).astype(np.float32)

        self.n_features_out_ = Xs.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        Xs = self.scaler.transform(X).astype(np.float32)
        if self.svd is not None:
            Xs = self.svd.transform(Xs).astype(np.float32)
        elif self.rp_matrix_ is not None:
            Xs = (Xs @ self.rp_matrix_).astype(np.float32)
        return Xs

    def fit_transform(self, X_train: np.ndarray, seed: int = 42) -> np.ndarray:
        self.fit(X_train, seed=seed)
        return self.transform(X_train)

    def summary(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "n_features_out": self.n_features_out_,
            "use_svd": self.svd is not None,
            "use_random_projection": self.rp_matrix_ is not None,
            "policy": None if self.policy_ is None else self.policy_.__dict__,
        }


def apply_modality_encoder(
    X_train: np.ndarray,
    X_test: np.ndarray,
    category: str,
    config: Dict[str, Any],
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Apply MCE when adadae.use_mce is enabled and category matches."""
    adadae_cfg = config.get("adadae", {})
    if not adadae_cfg.get("use_mce", False):
        return X_train, X_test, {"use_mce": False}

    mce_modality = str(adadae_cfg.get("mce_modality", category))
    if setting_blocks_mce(config, category):
        return X_train, X_test, {"use_mce": False, "blocked": True}

    enc = ModalityEncoder(modality=mce_modality)
    X_train = enc.fit_transform(X_train, seed=seed)
    X_test = enc.transform(X_test)
    summary = enc.summary()
    summary["use_mce"] = True
    return X_train, X_test, summary


def setting_blocks_mce(config: Dict[str, Any], category: str) -> bool:
    """Semi NLP: historical regressions with extra transforms."""
    adadae_cfg = config.get("adadae", {})
    block_semi_nlp = adadae_cfg.get("mce_block_semi_nlp", True)
    setting = str(config.get("_setting", ""))
    if block_semi_nlp and setting == "semi-supervised" and category == "nlp":
        return True
    return False
