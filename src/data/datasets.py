"""ADBench dataset registry and loaders for the 57-dataset protocol."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


CLASSICAL_FILES = [
    "1_ALOI.npz",
    "2_annthyroid.npz",
    "3_backdoor.npz",
    "4_breastw.npz",
    "5_campaign.npz",
    "6_cardio.npz",
    "7_Cardiotocography.npz",
    "8_celeba.npz",
    "9_census.npz",
    "10_cover.npz",
    "11_donors.npz",
    "12_fault.npz",
    "13_fraud.npz",
    "14_glass.npz",
    "15_Hepatitis.npz",
    "16_http.npz",
    "17_InternetAds.npz",
    "18_Ionosphere.npz",
    "19_landsat.npz",
    "20_letter.npz",
    "21_Lymphography.npz",
    "22_magic.gamma.npz",
    "23_mammography.npz",
    "24_mnist.npz",
    "25_musk.npz",
    "26_optdigits.npz",
    "27_PageBlocks.npz",
    "28_pendigits.npz",
    "29_Pima.npz",
    "30_satellite.npz",
    "31_satimage-2.npz",
    "32_shuttle.npz",
    "33_skin.npz",
    "34_smtp.npz",
    "35_SpamBase.npz",
    "36_speech.npz",
    "37_Stamps.npz",
    "38_thyroid.npz",
    "39_vertebral.npz",
    "40_vowels.npz",
    "41_Waveform.npz",
    "42_WBC.npz",
    "43_WDBC.npz",
    "44_Wilt.npz",
    "45_wine.npz",
    "46_WPBC.npz",
    "47_yeast.npz",
]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    category: str  # classical | cv | nlp
    relative_paths: Tuple[str, ...]


def _glob_prefix(directory: Path, prefix: str) -> Tuple[str, ...]:
    files = sorted(p.name for p in directory.glob(f"{prefix}*.npz"))
    return tuple(f"{directory.name}/{f}" for f in files)


def build_registry(adbench_root: Path) -> List[DatasetSpec]:
    root = Path(adbench_root)
    classical = root / "Classical"
    cv = root / "CV_by_ResNet18"
    nlp = root / "NLP_by_BERT"

    specs: List[DatasetSpec] = []
    for fname in CLASSICAL_FILES:
        name = fname.replace(".npz", "").split("_", 1)[1] if "_" in fname else fname
        # Prefer human names matching paper table where possible
        specs.append(
            DatasetSpec(
                name=_classical_display_name(fname),
                category="classical",
                relative_paths=(f"Classical/{fname}",),
            )
        )

    # CV families (average over splits at eval time)
    for family in ["CIFAR10", "FashionMNIST", "MNIST-C", "MVTec-AD", "SVHN"]:
        paths = _glob_prefix(cv, family)
        if paths:
            specs.append(DatasetSpec(name=family, category="cv", relative_paths=paths))

    # NLP families
    nlp_map = {
        "Agnews": "agnews",
        "Amazon": "amazon",
        "Imdb": "imdb",
        "Yelp": "yelp",
        "20newsgroups": "20news",
    }
    for display, prefix in nlp_map.items():
        paths = _glob_prefix(nlp, prefix)
        if not paths:
            # single file without trailing underscore variants
            single = nlp / f"{prefix}.npz"
            if single.exists():
                paths = (f"NLP_by_BERT/{prefix}.npz",)
        if paths:
            specs.append(DatasetSpec(name=display, category="nlp", relative_paths=paths))

    return specs


def _classical_display_name(fname: str) -> str:
    stem = fname.replace(".npz", "")
    # strip leading index: "6_cardio" -> "cardio"
    if "_" in stem and stem.split("_", 1)[0].isdigit():
        return stem.split("_", 1)[1]
    return stem


def load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"]).astype(np.int64).ravel()
    return X, y


def load_dataset_files(
    adbench_root: Path, relative_paths: Sequence[str]
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    root = Path(adbench_root)
    out = []
    for rel in relative_paths:
        X, y = load_npz(root / rel)
        out.append((rel, X, y))
    return out


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    train_setting: str = "unsupervised",
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Paper-faithful split (AnoDDAE / Livernoche protocol)."""
    if random_state is not None:
        rng = np.random.RandomState(random_state)
    else:
        rng = np.random.RandomState()

    if train_setting == "unsupervised":
        return X, X, y, y

    if train_setting == "semi-supervised":
        anomaly_indices = np.where(y == 1)[0]
        normal_indices = np.where(y == 0)[0]
        rng.shuffle(normal_indices)
        half = len(normal_indices) // 2
        train_idx = normal_indices[:half]
        test_idx = np.concatenate([normal_indices[half:], anomaly_indices])
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    raise ValueError("train_setting must be 'unsupervised' or 'semi-supervised'")


def carve_val_from_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    val_fraction: float = 0.2,
    random_state: Optional[int] = None,
    min_val: int = 8,
    min_fit: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Hold out a validation slice from train only (never from test).

    Used for early stopping / checkpointing without test labels.
    Returns (X_fit, X_val, y_fit, y_val).
    """
    if val_fraction <= 0.0:
        return X_train, X_train[:0], y_train, y_train[:0]

    n = int(X_train.shape[0])
    if n < (min_val + min_fit):
        # Too small to carve — train on all; caller should fall back to train loss.
        return X_train, X_train[:0], y_train, y_train[:0]

    rng = np.random.RandomState(random_state if random_state is not None else 0)
    n_val = int(round(n * float(val_fraction)))
    n_val = max(min_val, min(n_val, n - min_fit))
    idx = rng.permutation(n)
    val_idx = idx[:n_val]
    fit_idx = idx[n_val:]
    return (
        X_train[fit_idx],
        X_train[val_idx],
        y_train[fit_idx],
        y_train[val_idx],
    )


def list_dataset_names(adbench_root: Path) -> List[str]:
    return [s.name for s in build_registry(adbench_root)]


def get_spec(adbench_root: Path, name: str) -> DatasetSpec:
    for spec in build_registry(adbench_root):
        if spec.name.lower() == name.lower() or spec.name == name:
            return spec
    # also allow classical file stem
    for spec in build_registry(adbench_root):
        if any(name in p for p in spec.relative_paths):
            return spec
    raise KeyError(f"Unknown dataset: {name}")
