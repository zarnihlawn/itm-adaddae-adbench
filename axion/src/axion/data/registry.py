"""ADBench 57-dataset registry (Classical + ResNet18 + BERT)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from axion.paths import DEFAULT_ADBENCH_DATASETS

CLASSICAL_FILES: Tuple[str, ...] = (
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
)

# Paper Table 3 category labels (AnoDDAE Appendix B.1)
PAPER_CATEGORIES: Dict[str, str] = {
    "ALOI": "Image",
    "annthyroid": "Healthcare",
    "backdoor": "Network",
    "breastw": "Healthcare",
    "campaign": "Finance",
    "cardio": "Healthcare",
    "Cardiotocography": "Healthcare",
    "celeba": "Image",
    "census": "Sociology",
    "cover": "Botany",
    "donors": "Sociology",
    "fault": "Physical",
    "fraud": "Finance",
    "glass": "Forensic",
    "Hepatitis": "Healthcare",
    "http": "Web",
    "InternetAds": "Image",
    "Ionosphere": "Oryctognosy",
    "landsat": "Astronautics",
    "letter": "Image",
    "Lymphography": "Healthcare",
    "magic.gamma": "Physical",
    "mammography": "Healthcare",
    "mnist": "Image",
    "musk": "Chemistry",
    "optdigits": "Image",
    "PageBlocks": "Document",
    "pendigits": "Image",
    "Pima": "Healthcare",
    "satellite": "Astronautics",
    "satimage-2": "Astronautics",
    "shuttle": "Astronautics",
    "skin": "Image",
    "smtp": "Web",
    "SpamBase": "Document",
    "speech": "Linguistics",
    "Stamps": "Document",
    "thyroid": "Healthcare",
    "vertebral": "Biology",
    "vowels": "Linguistics",
    "Waveform": "Physics",
    "WBC": "Healthcare",
    "WDBC": "Healthcare",
    "Wilt": "Botany",
    "wine": "Chemistry",
    "WPBC": "Healthcare",
    "yeast": "Biology",
    "CIFAR10": "Image",
    "FashionMNIST": "Image",
    "MNIST-C": "Image",
    "MVTec-AD": "Image",
    "SVHN": "Image",
    "Agnews": "NLP",
    "Amazon": "NLP",
    "Imdb": "NLP",
    "Yelp": "NLP",
    "20newsgroups": "NLP",
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    modality: str  # classical | cv | nlp
    paper_category: str
    relative_paths: Tuple[str, ...]
    embedding: str  # none | ResNet18 | BERT


def _classical_display_name(fname: str) -> str:
    stem = fname.replace(".npz", "")
    if "_" in stem and stem.split("_", 1)[0].isdigit():
        return stem.split("_", 1)[1]
    return stem


def _glob_prefix(directory: Path, folder_name: str, prefix: str) -> Tuple[str, ...]:
    files = sorted(p.name for p in directory.glob(f"{prefix}*.npz"))
    return tuple(f"{folder_name}/{f}" for f in files)


def build_registry(adbench_root: Optional[Path] = None) -> List[DatasetSpec]:
    root = Path(adbench_root) if adbench_root is not None else DEFAULT_ADBENCH_DATASETS
    classical = root / "Classical"
    cv = root / "CV_by_ResNet18"
    nlp = root / "NLP_by_BERT"

    specs: List[DatasetSpec] = []
    for fname in CLASSICAL_FILES:
        name = _classical_display_name(fname)
        specs.append(
            DatasetSpec(
                name=name,
                modality="classical",
                paper_category=PAPER_CATEGORIES.get(name, "Classical"),
                relative_paths=(f"Classical/{fname}",),
                embedding="none",
            )
        )

    for family in ("CIFAR10", "FashionMNIST", "MNIST-C", "MVTec-AD", "SVHN"):
        paths = _glob_prefix(cv, "CV_by_ResNet18", family)
        if not paths:
            raise FileNotFoundError(f"Missing CV family {family} under {cv}")
        specs.append(
            DatasetSpec(
                name=family,
                modality="cv",
                paper_category=PAPER_CATEGORIES[family],
                relative_paths=paths,
                embedding="ResNet18",
            )
        )

    nlp_map = {
        "Agnews": "agnews",
        "Amazon": "amazon",
        "Imdb": "imdb",
        "Yelp": "yelp",
        "20newsgroups": "20news",
    }
    for display, prefix in nlp_map.items():
        paths = _glob_prefix(nlp, "NLP_by_BERT", prefix)
        if not paths:
            single = nlp / f"{prefix}.npz"
            if single.exists():
                paths = (f"NLP_by_BERT/{prefix}.npz",)
        if not paths:
            raise FileNotFoundError(f"Missing NLP family {display} under {nlp}")
        specs.append(
            DatasetSpec(
                name=display,
                modality="nlp",
                paper_category=PAPER_CATEGORIES[display],
                relative_paths=paths,
                embedding="BERT",
            )
        )

    if len(specs) != 57:
        raise RuntimeError(f"Expected 57 datasets, got {len(specs)}")
    return specs


def registry_by_name(adbench_root: Optional[Path] = None) -> Dict[str, DatasetSpec]:
    return {s.name: s for s in build_registry(adbench_root)}


def load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"]).astype(np.int64).ravel()
    if set(np.unique(y)) - {0, 1}:
        raise ValueError(f"Non-binary labels in {path}: {np.unique(y)}")
    return X, y


def load_dataset_files(
    relative_paths: Sequence[str],
    adbench_root: Optional[Path] = None,
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    root = Path(adbench_root) if adbench_root is not None else DEFAULT_ADBENCH_DATASETS
    out: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for rel in relative_paths:
        X, y = load_npz(root / rel)
        out.append((rel, X, y))
    return out
