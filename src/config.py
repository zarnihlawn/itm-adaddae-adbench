"""Config loading and path helpers."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # project/

# Short names / tiers -> configs/*.yaml (relative to configs/)
HARDWARE_PROFILES: Dict[str, str] = {
    "8gb": "hardware_rtx5070.yaml",
    "12gb": "hardware_rtx5070_12g.yaml",
    "16gb": "hardware_rtx5070ti.yaml",
    "rtx5070": "hardware_rtx5070.yaml",
    "rtx5060": "hardware_rtx5070.yaml",
    "rtx5070_8g": "hardware_rtx5070.yaml",
    "rtx5070_12g": "hardware_rtx5070_12g.yaml",
    "rtx5070ti": "hardware_rtx5070ti.yaml",
    "rtx5060ti": "hardware_rtx5070ti.yaml",
}


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_hardware_name(name: str) -> str:
    key = name.strip().lower()
    if key in HARDWARE_PROFILES:
        return HARDWARE_PROFILES[key]
    if not key.endswith(".yaml"):
        key = f"{key}.yaml" if not Path(key).suffix else key
    return key


def hardware_yaml_path(hw_name: str, config_dir: Path) -> Path:
    resolved = resolve_hardware_name(hw_name)
    hw_path = config_dir / resolved
    if hw_path.exists():
        return hw_path
    fallback = PROJECT_ROOT / "configs" / Path(resolved).name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Hardware profile not found: {hw_name} -> {resolved}")


def detect_vram_gb() -> Optional[float]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        mb = float(out.strip().split("\n")[0])
        return mb / 1024.0
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


def suggest_hardware_profile() -> str:
    """Pick hardware YAML from detected VRAM (falls back to 16 GB default)."""
    vram = detect_vram_gb()
    if vram is None:
        return "hardware_rtx5070ti.yaml"
    if vram >= 15.0:
        return "hardware_rtx5070ti.yaml"
    if vram >= 11.0:
        return "hardware_rtx5070_12g.yaml"
    return "hardware_rtx5070.yaml"


def apply_hardware_override(cfg: Dict[str, Any], hardware: str, config_path: Path) -> Dict[str, Any]:
    hw_path = hardware_yaml_path(hardware, config_path.parent)
    cfg["hardware"] = load_yaml(hw_path)
    return cfg


# Canonical ADBench datasets path (sibling repo next to this project checkout).
# Vast: /workspace/ITM/ADBench/adbench/datasets  ↔  ../ADBench/adbench/datasets
DEFAULT_ADBENCH_ROOT = "../ADBench/adbench/datasets"


def resolve_adbench_root(configured: str | None = None) -> Path:
    """Resolve ADBench datasets dir relative to repo root (this project checkout)."""
    rel = configured or DEFAULT_ADBENCH_ROOT
    path = Path(rel)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    if path.is_dir():
        return path

    raise FileNotFoundError(
        f"ADBench datasets not found at {path}. "
        f"Expected sibling layout: <parent>/ADBench/adbench/datasets next to "
        f"<parent>/project (repo root). On Vast: /workspace/ITM/ADBench/adbench/datasets. "
        f"Override with paths.adbench_root in config."
    )


def load_config(config_path: str | Path, hardware: str | None = None) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    cfg = load_yaml(path)

    if hardware:
        cfg = apply_hardware_override(cfg, hardware, path)
    else:
        hw_name = cfg.get("hardware", "hardware_cpu.yaml")
        hw_path = path.parent / hw_name if not Path(hw_name).is_absolute() else Path(hw_name)
        if not hw_path.exists():
            hw_path = PROJECT_ROOT / "configs" / Path(hw_name).name
        hw = load_yaml(hw_path)
        cfg["hardware"] = hw

    paths = cfg.setdefault("paths", {})
    adbench = paths.get("adbench_root")
    paths["adbench_root"] = str(resolve_adbench_root(adbench))

    results = paths.get("results_dir", "results")
    results_path = Path(results)
    if not results_path.is_absolute():
        results_path = PROJECT_ROOT / results_path
    paths["results_dir"] = str(results_path)
    return cfg
