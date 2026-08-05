#!/usr/bin/env python3
"""Loop 5: verify semi baseline scoring matches AnoDDAE full-sum (t=1..T-1).

Checks BASELINE_DDAE / PER-resolved baseline policies have use_scs=False or
scs_mode=full_sum so score_timesteps == range(1, T).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_yaml  # noqa: E402
from src.policy import BASELINE_DDAE, policy_overrides  # noqa: E402
from src.policy_per import apply_per_config  # noqa: E402


def _ok_full_sum(adadae: dict) -> bool:
    if adadae.get("scs_full_sum_ablation"):
        return True
    if adadae.get("scs_mode") == "full_sum":
        return True
    if not adadae.get("use_scs", True):
        return True
    return False


def main() -> int:
    cfg = load_yaml(PROJECT_ROOT / "configs" / "adadae_per.yaml")
    errs = []
    if not _ok_full_sum(BASELINE_DDAE.get("adadae", {})):
        errs.append("BASELINE_DDAE does not force full-sum / SCS-off")

    for ds, cat in [
        ("cardio", "classical"),
        ("Wilt", "classical"),
        ("glass", "classical"),
        ("CIFAR10", "cv"),
        ("Agnews", "nlp"),
    ]:
        out = apply_per_config(cfg, "semi-supervised", cat, ds, meta={"n": 500, "d": 20})
        ad = out.get("adadae", {})
        pol = ad.get("resolved_policy", "")
        if "baseline" in pol or "nlp_frozen" in pol or "nlp_baseline" in pol:
            if not _ok_full_sum(ad):
                errs.append(f"{ds} pol={pol} not full-sum: {ad.get('use_scs')=} {ad.get('scs_mode')=}")

    # Instantiated model path check for use_scs=False
    from src.models.danc import NoiseConfig
    from src.models.adadae import AdaDDAE
    import torch

    noise = NoiseConfig(50, "linear", 1e-4, 0.02, 4)
    m = AdaDDAE(
        input_dim=8,
        noise_config=noise,
        use_scs=False,
        scs_mode="full_sum",
        use_multiview=False,
        use_dte_view=False,
        use_rejection_training=False,
        device=torch.device("cpu"),
        epochs=1,
    )
    expected = list(range(1, 50))
    if list(m.score_timesteps) != expected:
        errs.append(f"AdaDDAE full_sum timesteps {m.score_timesteps[:5]}... != range(1,50)")

    report = {
        "pass": len(errs) == 0,
        "errors": errs,
        "ano_ddae_predict": "sum L2 over t=1..T-1 (AnoDDAE/src/model.py predict)",
        "note": "PER baseline_ddae / nlp_frozen paths force use_scs=False + scs_full_sum_ablation",
    }
    out_path = PROJECT_ROOT / "results/adadae_per/thesis/loop5_scoring_parity.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
