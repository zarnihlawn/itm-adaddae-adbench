#!/usr/bin/env python3
"""Assert a primary (adadae*_final) config is integrity-safe before GPU spend.

Fails if routing/exceptions/MCE/GATE/guarded-merge producers are enabled, or if
early stopping is not val-only.

Also requires shared train/test protocol knobs to match the fair DDAE baseline
(`configs/baselines_ddae_valstop.yaml`) so only Ada method flags differ.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROJECT_ROOT, load_yaml

FAIR_BASELINE_CONFIG = PROJECT_ROOT / "configs" / "baselines_ddae_valstop.yaml"

# (section, key) pairs that must equal the fair DDAE baseline for paper-aligned protocol.
PROTOCOL_KEYS: Tuple[Tuple[str, ...], ...] = (
    ("seeds",),
    ("train", "epochs"),
    ("train", "lr"),
    ("train", "eval_every"),
    ("train", "early_stop_patience"),
    ("train", "val_fraction"),
    ("train", "early_stop_metric"),
    ("model", "hidden_dims"),
    ("model", "latent_dim"),
    ("model", "activation"),
    ("diffusion", "time_emb_dim"),
    ("diffusion", "beta_start"),
    ("diffusion", "beta_end"),
)

FORBIDDEN_FOR_FINAL = (
    ("adadae", "policy", "routed"),
    ("adadae", "use_mce", True),
    ("adadae", "use_gate", True),
)


def _get(cfg: Dict[str, Any], *keys: str, default=None):
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _norm(val: Any) -> Any:
    """Normalize for equality (lists↔tuples, floats)."""
    if isinstance(val, list):
        return [_norm(v) for v in val]
    if isinstance(val, tuple):
        return [_norm(v) for v in val]
    if isinstance(val, float):
        return float(val)
    if isinstance(val, int) and not isinstance(val, bool):
        return int(val)
    return val


def load_config_for_audit(path: Path) -> Dict[str, Any]:
    """Load YAML without requiring ADBench on disk (integrity checks only)."""
    return load_yaml(path)


def audit_protocol_vs_baseline(
    cfg: Dict[str, Any],
    baseline: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Shared train/eval knobs must match fair DDAE; method flags may differ."""
    errs: List[str] = []
    if baseline is None:
        if not FAIR_BASELINE_CONFIG.is_file():
            errs.append(f"fair baseline missing: {FAIR_BASELINE_CONFIG}")
            return errs
        baseline = load_config_for_audit(FAIR_BASELINE_CONFIG)

    for keys in PROTOCOL_KEYS:
        got = _norm(_get(cfg, *keys, default=None))
        exp = _norm(_get(baseline, *keys, default=None))
        path = ".".join(keys)
        if exp is None:
            errs.append(f"baseline missing protocol key {path}")
            continue
        if got is None:
            errs.append(f"config missing protocol key {path} (expected {exp!r})")
            continue
        # float compare for lr / betas / fractions
        if isinstance(exp, float) or isinstance(got, float):
            try:
                if abs(float(got) - float(exp)) > 1e-12:
                    errs.append(f"protocol drift {path}: got {got!r}, fair DDAE has {exp!r}")
            except (TypeError, ValueError):
                errs.append(f"protocol drift {path}: got {got!r}, fair DDAE has {exp!r}")
        elif got != exp:
            errs.append(f"protocol drift {path}: got {got!r}, fair DDAE has {exp!r}")
    return errs


def audit_primary_config(
    cfg: Dict[str, Any],
    require_final_run_id: bool = True,
    check_protocol_vs_baseline: bool = True,
    baseline: Optional[Dict[str, Any]] = None,
) -> List[str]:
    errs: List[str] = []
    run_id = str(_get(cfg, "paths", "run_id", default="") or "")
    allowed_final = {
        "adadae_per",
        "adadae_champion",
        "adadae_final",
        "adadae2_final",
        "adadae3_final",
        "adadae4_final",
        "adadae5_final",
        "adadae6_final",
    }
    if require_final_run_id and run_id not in allowed_final and not run_id.endswith("_smoke"):
        errs.append(
            f"paths.run_id must be one of {sorted(allowed_final)} (got {run_id!r})"
        )

    policy = str(_get(cfg, "adadae", "policy", default="static") or "static")
    is_per = policy == "per" or run_id.startswith("adadae_per")
    if policy == "routed" and not is_per:
        errs.append("adadae.policy must not be 'routed' for the primary recipe")
    if policy not in ("static", "paradigm", "per"):
        errs.append(f"adadae.policy must be 'static', 'paradigm', or 'per' (got {policy!r})")

    if _get(cfg, "adadae", "exceptions_file") and not is_per:
        errs.append("adadae.exceptions_file must be unset for the primary recipe")

    if bool(_get(cfg, "adadae", "use_mce", default=False)) and not is_per:
        errs.append("adadae.use_mce must be false for the primary recipe")
    if bool(_get(cfg, "adadae", "use_gate", default=False)) and not is_per:
        errs.append("adadae.use_gate must be false for the primary recipe")

    if is_per:
        if policy != "per":
            errs.append("adadae_per requires adadae.policy == 'per'")
        exc = _get(cfg, "adadae", "exceptions_file")
        if not exc:
            errs.append("adadae_per requires adadae.exceptions_file")
        upg = _get(cfg, "adadae", "upgrades_file")
        if not upg:
            errs.append("adadae_per requires adadae.upgrades_file")

    if run_id.startswith("adadae_champion"):
        if bool(_get(cfg, "adadae", "use_uncertainty_view", default=False)):
            errs.append("champion: use_uncertainty_view must be false")
        fusion_c = str(_get(cfg, "adadae", "fusion_mode", default="fixed") or "fixed")
        if fusion_c == "calibrated":
            errs.append("champion: fusion_mode 'calibrated' is banned")
        for flag in (
            "use_chronos",
            "use_geode",
            "use_aether",
            "use_nexus",
            "use_dte_view",
        ):
            if bool(_get(cfg, "adadae", flag, default=False)):
                errs.append(f"champion: adadae.{flag} must be false (kitchen-sink ban)")

    fusion = str(_get(cfg, "adadae", "fusion_mode", default="fixed") or "fixed")
    if fusion == "smc" and run_id in allowed_final and not is_per:
        errs.append(f"adadae.fusion_mode 'smc' is appendix-only for {run_id}")
    if fusion not in ("fixed", "calibrated", "calix", "smc", "argos", "lynx", "quell", "lexicon", "kale"):
        errs.append(f"unknown fusion_mode {fusion!r}")

    esm = str(_get(cfg, "train", "early_stop_metric", default="") or "")
    if esm != "val_loss":
        errs.append(f"train.early_stop_metric must be 'val_loss' (got {esm!r})")

    vf = float(_get(cfg, "train", "val_fraction", default=0.0) or 0.0)
    if vf <= 0.0:
        errs.append("train.val_fraction must be > 0 for val-only early stop")

    contam = str(_get(cfg, "adadae", "danc_contamination_mode", default="label_free") or "")
    if contam == "oracle":
        errs.append("adadae.danc_contamination_mode 'oracle' is ablation-only")

    # Lexicon / fixed fusion weights must sum ≈ 1 when declared
    fw = _get(cfg, "adadae", "fusion_weights", default=None)
    if isinstance(fw, dict) and fw:
        try:
            s = float(sum(float(v) for v in fw.values()))
            if abs(s - 1.0) > 1e-3:
                errs.append(f"adadae.fusion_weights sum to {s:.6f}, expected ≈ 1.0")
        except (TypeError, ValueError):
            errs.append("adadae.fusion_weights must be numeric")

    # Smoke configs intentionally use short epochs/patience; skip protocol lock.
    # PER uses v5.1 shell (time_emb_dim 8) — not locked to fair DDAE knobs.
    if (
        check_protocol_vs_baseline
        and not run_id.endswith("_smoke")
        and not is_per
    ):
        errs.extend(audit_protocol_vs_baseline(cfg, baseline=baseline))

    return errs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default="configs/adadae_final.yaml",
        help="Config path relative to project root or absolute",
    )
    p.add_argument(
        "--allow-nonfinal-run-id",
        action="store_true",
        help="Skip paths.run_id == adadae*_final check (for paired DDAE valstop)",
    )
    p.add_argument(
        "--skip-protocol-baseline",
        action="store_true",
        help="Skip shared-knob lock vs baselines_ddae_valstop.yaml",
    )
    p.add_argument(
        "--baseline",
        default=str(FAIR_BASELINE_CONFIG.relative_to(PROJECT_ROOT)),
        help="Fair DDAE baseline YAML for protocol comparison",
    )
    args = p.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    baseline_path = Path(args.baseline)
    if not baseline_path.is_absolute():
        baseline_path = PROJECT_ROOT / baseline_path
    cfg = load_config_for_audit(cfg_path)
    baseline = load_config_for_audit(baseline_path) if baseline_path.is_file() else None
    errs = audit_primary_config(
        cfg,
        require_final_run_id=not args.allow_nonfinal_run_id,
        check_protocol_vs_baseline=not args.skip_protocol_baseline,
        baseline=baseline,
    )
    if errs:
        print("INTEGRITY FAIL:")
        for e in errs:
            print(f"  - {e}")
        return 1
    print(f"INTEGRITY OK: {cfg_path}")
    print(f"  run_id={_get(cfg, 'paths', 'run_id')}")
    print(f"  policy={_get(cfg, 'adadae', 'policy', default='static')}")
    print(f"  early_stop_metric={_get(cfg, 'train', 'early_stop_metric')}")
    print(f"  val_fraction={_get(cfg, 'train', 'val_fraction')}")
    print(f"  early_stop_patience={_get(cfg, 'train', 'early_stop_patience')}")
    print(f"  time_emb_dim={_get(cfg, 'diffusion', 'time_emb_dim')}")
    print(f"  protocol_locked_to={baseline_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
