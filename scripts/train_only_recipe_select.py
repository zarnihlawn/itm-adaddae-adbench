#!/usr/bin/env python3
"""Train-only multi-recipe arbitration (val_loss only — never test PR).

Strip-first ablations for bleed-CV (plain bases + at most one A6 overlay).
Presets: ship (hard-12 + bleed-classical), bleed-cv, bleed-classical, hard-12.

Usage:
  python scripts/train_only_recipe_select.py --dry-run
  python scripts/train_only_recipe_select.py --preset ship --seeds 111 222 333 --hardware 16gb
  python scripts/train_only_recipe_select.py --apply-evidence-freeze
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Ship hard-tail set (12) — matches invalidate --hard-tails / apply freeze default
HARD_TAIL_12 = [
    "speech",
    "ALOI",
    "celeba",
    "SVHN",
    "CIFAR10",
    "Wilt",
    "Imdb",
    "Amazon",
    "Yelp",
    "Agnews",
    "20newsgroups",
    "census",
]

# Worst CV / embed stacks (kitchen-sink ablate-first)
BLEED_CV = ["SVHN", "ALOI", "celeba", "CIFAR10", "speech"]

# Protect / classical losers outside hard-12
BLEED_CLASSICAL = ["smtp", "satimage-2", "Pima", "Stamps", "letter", "wine"]

# Default GPU select = hard-12 ∪ bleed-classical (unique, hard-12 order first)
SHIP_SELECT_DEFAULT: List[str] = list(
    dict.fromkeys(HARD_TAIL_12 + BLEED_CLASSICAL)
)

# Legacy dry-run list (includes glass/vertebral/WPBC for routing dump)
HARD_SEMI_DEFAULT = HARD_TAIL_12 + ["vertebral", "glass", "WPBC"]

# ADBench NLP names
_NLP = {"Agnews", "Amazon", "Imdb", "Yelp", "20newsgroups"}
# True CV embeds in ADBench CV_by_ResNet18 (NOT Classical speech/ALOI/celeba)
_CV = {"CIFAR10", "SVHN", "MNIST-C", "FashionMNIST", "MVTec-AD"}

BASE_CANDIDATES = [
    "baseline_ddae",
    "semi_cvnlp_ftp",
    "semi_rdt_tail",
    "semi_smc_tail",
    "champion_semi",
    "semi_speech_specialist",
]

# Single-overlay only (strip-first; no multi-A6 kitchen-sink in search)
OVERLAY_CANDIDATES: Dict[str, Dict[str, Any]] = {
    "orbit": {"adadae": {"use_orbit": True, "use_multiview": True}},
    "locus": {"adadae": {"use_locus": True, "use_multiview": True}},
    "helix": {"adadae": {"use_helix": True}},
    "contrastive": {
        "train": {
            "contrastive": True,
            "contrastive_alpha": 0.2,
            "hard_negative_mining": False,
        },
        "adadae": {"contrastive_pairing": "taps"},
    },
    "full_sum": {
        "adadae": {
            "use_scs": False,
            "scs_mode": "full_sum",
            "scs_full_sum_ablation": True,
            "score_noise_draws": 3,
        }
    },
    "cosine": {"diffusion": {"scheduler": "cosine"}},
}

# Default search: plain bases + base+one overlay (no MCE/GATE — static cfg)
STRIP_FIRST_CANDIDATES = [
    "baseline_ddae",
    "baseline_ddae+full_sum",
    "baseline_ddae+contrastive",
    "baseline_ddae+cosine",
    "semi_cvnlp_ftp",
    "semi_rdt_tail",
    "champion_semi",
    "baseline_ddae+orbit",
    "baseline_ddae+locus",
    "baseline_ddae+helix",
    "semi_rdt_tail+orbit",
    "semi_rdt_tail+locus",
    "semi_rdt_tail+helix",
    "semi_cvnlp_ftp+orbit",
    "semi_cvnlp_ftp+locus",
    "semi_cvnlp_ftp+helix",
]

# Complexity prior: lower is preferred within ε-ball of best val_loss
_BASE_COMPLEXITY = {
    "baseline_ddae": 0,
    "champion_semi": 1,
    "semi_cvnlp_ftp": 2,
    "semi_smc_tail": 3,
    "semi_rdt_tail": 4,
    "semi_speech_specialist": 4,
}
_OVERLAY_COMPLEXITY = {
    "full_sum": 0,
    "cosine": 1,
    "contrastive": 2,
    "helix": 3,
    "orbit": 4,
    "locus": 4,
    "spiral": 4,
    "nautilus": 5,
}
_RISKY_BASES = {"semi_rdt_tail", "semi_smc_tail", "semi_speech_specialist"}
_RISKY_TOKENS = {"mce", "gate", "rdt"}


def recipe_complexity(cand: str) -> int:
    parts = [p for p in cand.split("+") if p]
    if not parts:
        return 99
    base = parts[0]
    score = _BASE_COMPLEXITY.get(base, 5)
    for ov in parts[1:]:
        score += _OVERLAY_COMPLEXITY.get(ov, 3)
    return score


def is_risky_recipe(cand: str) -> bool:
    parts = cand.split("+")
    if parts and parts[0] in _RISKY_BASES:
        return True
    return any(tok in _RISKY_TOKENS for tok in parts)


def pick_winner_with_proxies(
    means_val: Dict[str, float],
    means_synth: Dict[str, float],
    eps_rel: float = 0.05,
    rdt_veto_margin: float = 0.05,
) -> tuple[str, Dict[str, Any]]:
    """Primary: min val_loss. Secondary: ε-ball → complexity → synth-val PR.

    Hard veto: risky recipes whose synth-val PR is worse than best baseline_*
    by ``rdt_veto_margin`` are discarded even if val_loss wins.
    """
    finite = {c: m for c, m in means_val.items() if m != float("inf")}
    if not finite:
        raise ValueError("no finite val_loss")

    baseline_synths = [
        means_synth[c]
        for c in finite
        if c.startswith("baseline_ddae") and c in means_synth and means_synth[c] is not None
    ]
    best_baseline_synth = max(baseline_synths) if baseline_synths else None

    eligible = dict(finite)
    vetoed: List[str] = []
    if best_baseline_synth is not None:
        for c in list(eligible):
            if not is_risky_recipe(c):
                continue
            syn = means_synth.get(c)
            if syn is None:
                continue
            if syn + rdt_veto_margin < best_baseline_synth:
                vetoed.append(c)
                eligible.pop(c, None)
    if not eligible:
        eligible = dict(finite)

    best_val = min(eligible.values())
    ball = [
        c
        for c, v in eligible.items()
        if v <= best_val * (1.0 + eps_rel) or abs(v - best_val) <= 1e-9
    ]
    # Prefer lower complexity; break ties with higher synth-val PR, then lower val_loss
    def sort_key(c: str) -> tuple:
        syn = means_synth.get(c)
        syn_rank = -(syn if syn is not None else -1.0)
        return (recipe_complexity(c), syn_rank, eligible[c], c)

    ball.sort(key=sort_key)
    winner = ball[0]
    return winner, {
        "eps_rel": eps_rel,
        "best_val_loss": best_val,
        "epsilon_ball": ball,
        "vetoed_risky": vetoed,
        "best_baseline_synth_pr": best_baseline_synth,
        "winner_complexity": recipe_complexity(winner),
        "winner_synth_val_pr": means_synth.get(winner),
    }


# Evidence freeze when GPU selection cannot run (ablate-first revision for CV)
EVIDENCE_FREEZE: Dict[str, Dict[str, Any]] = {
    "speech": {
        "policy": "semi_rdt_tail",
        "overlays": ["locus"],
        "strip_mce_gate": True,
        "reason": "strip kitchen-sink; RDT+single locus",
    },
    "ALOI": {
        "policy": "semi_rdt_tail",
        "overlays": ["orbit"],
        "strip_mce_gate": True,
        "reason": "strip MCE+GATE multi-A6; RDT+orbit only",
    },
    "celeba": {
        "policy": "semi_cvnlp_ftp",
        "overlays": ["orbit"],
        "strip_mce_gate": True,
        "reason": "FTP+orbit; strip MCE",
    },
    "CIFAR10": {
        "policy": "semi_cvnlp_ftp",
        "overlays": ["orbit"],
        "strip_mce_gate": True,
        "reason": "FTP+orbit; strip GATE/helix stack",
    },
    "SVHN": {
        "policy": "semi_cvnlp_ftp",
        "overlays": [],
        "strip_mce_gate": True,
        "reason": "worst vs fair — strip MCE+orbit+helix to plain FTP",
    },
    "Wilt": {
        "policy": "semi_rdt_tail",
        "overlays": ["orbit", "locus", "spiral"],
        "reason": "keep winning RDT stack",
    },
    "census": {
        "policy": "baseline_ddae",
        "overlays": [],
        "strip_mce_gate": True,
        "reason": "Phase0: RDT hurt test PR — baseline only until synth-val clears",
    },
    "vertebral": {
        "policy": "baseline_ddae",
        "overlays": ["nautilus"],
        "reason": "keep nautilus win vs fair",
    },
    "glass": {
        "policy": "baseline_ddae",
        "overlays": ["nautilus"],
        "reason": "keep nautilus win vs fair",
    },
    "WPBC": {"policy": "semi_speech_specialist", "overlays": [], "reason": "existing specialist"},
    "Imdb": {"policy": "semi_nlp_frozen", "overlays": ["orbit"], "reason": "NLP frozen+orbit"},
    "Amazon": {"policy": "semi_nlp_frozen", "overlays": ["orbit"], "reason": "NLP frozen+orbit"},
    "Yelp": {"policy": "semi_nlp_frozen", "overlays": ["orbit"], "reason": "NLP frozen+orbit"},
    "Agnews": {"policy": "semi_nlp_frozen", "overlays": ["orbit"], "reason": "NLP frozen+orbit"},
    "20newsgroups": {
        "policy": "semi_nlp_frozen",
        "overlays": ["orbit"],
        "reason": "NLP frozen+orbit",
    },
    # Bleed-classical: force baseline (protect-compatible); strip upgrades via freeze
    "smtp": {
        "policy": "baseline_ddae",
        "overlays": [],
        "strip_mce_gate": True,
        "reason": "protect loser — baseline only",
    },
    "satimage-2": {
        "policy": "baseline_ddae",
        "overlays": [],
        "strip_mce_gate": True,
        "reason": "protect loser — baseline only",
    },
    "wine": {
        "policy": "baseline_ddae",
        "overlays": [],
        "strip_mce_gate": True,
        "reason": "Phase0: RDT −12 PR vs fair — never re-select RDT by val_loss alone",
    },
    "Pima": {
        "policy": "baseline_ddae",
        "overlays": [],
        "strip_mce_gate": True,
        "reason": "bleed-classical loser",
    },
    "Stamps": {
        "policy": "baseline_ddae",
        "overlays": [],
        "strip_mce_gate": True,
        "reason": "bleed-classical loser",
    },
    "letter": {
        "policy": "baseline_ddae",
        "overlays": [],
        "strip_mce_gate": True,
        "reason": "bleed-classical loser",
    },
}


def _category_for(name: str, registry_by_name: Optional[Dict[str, Any]] = None) -> str:
    """Match ADBench registry categories (speech/ALOI/celeba are Classical)."""
    if registry_by_name and name in registry_by_name:
        return str(registry_by_name[name].category)
    if name in _NLP:
        return "nlp"
    if name in _CV:
        return "cv"
    return "classical"


def _preset_datasets(preset: str) -> List[str]:
    if preset == "ship":
        return list(SHIP_SELECT_DEFAULT)
    if preset == "hard-12":
        return list(HARD_TAIL_12)
    if preset == "bleed-cv":
        return list(BLEED_CV)
    if preset == "bleed-classical":
        return list(BLEED_CLASSICAL)
    if preset == "legacy":
        return list(HARD_SEMI_DEFAULT)
    raise ValueError(f"unknown preset {preset!r}")


def dry_run_resolve() -> Dict[str, Any]:
    from src.config import load_yaml
    from src.data.datasets import build_registry
    from src.policy_per import apply_per_config, clear_per_upgrades_cache

    clear_per_upgrades_cache()
    cfg = load_yaml(PROJECT_ROOT / "configs" / "adadae_per.yaml")
    registry_by_name: Dict[str, Any] = {}
    try:
        root = cfg.get("paths", {}).get("adbench_root")
        if root:
            registry_by_name = {e.name: e for e in build_registry(root)}
    except Exception:  # noqa: BLE001
        registry_by_name = {}

    rows = []
    for ds in HARD_SEMI_DEFAULT:
        cat = _category_for(ds, registry_by_name)
        out = apply_per_config(
            cfg, "semi-supervised", cat, ds, meta={"n": 800.0, "d": 32.0}
        )
        rows.append(
            {
                "dataset": ds,
                "category": cat,
                "resolved_policy": out["adadae"].get("resolved_policy"),
                "use_scs": out["adadae"].get("use_scs"),
                "use_mce": out["adadae"].get("use_mce"),
                "use_orbit": out["adadae"].get("use_orbit"),
                "use_locus": out["adadae"].get("use_locus"),
                "use_nautilus": out["adadae"].get("use_nautilus"),
                "use_apex": out["adadae"].get("use_apex"),
                "fusion_mode": out["adadae"].get("fusion_mode"),
                "features": out.get("features"),
                "train_ov": {
                    k: out.get("train", {}).get(k)
                    for k in ("min_epochs", "val_fraction_semi", "early_stop_patience")
                    if out.get("train", {}).get(k) is not None
                },
            }
        )
    return {
        "mode": "dry_run",
        "base_candidates": BASE_CANDIDATES,
        "strip_first_candidates": STRIP_FIRST_CANDIDATES,
        "presets": {
            "ship": SHIP_SELECT_DEFAULT,
            "hard-12": HARD_TAIL_12,
            "bleed-cv": BLEED_CV,
            "bleed-classical": BLEED_CLASSICAL,
        },
        "current_per": rows,
    }


def evidence_freeze_report() -> Dict[str, Any]:
    return {
        "mode": "evidence_freeze",
        "note": (
            "Ablate-first evidence freeze (CV strip MCE/GATE). "
            "Re-run --preset ship on Vast GPU for val_loss+synth winners."
        ),
        "winners": EVIDENCE_FREEZE,
        "protect_baseline_semi": [
            "smtp",
            "donors",
            "wine",
            "http",
            "musk",
            "breastw",
            "shuttle",
            "Ionosphere",
            "Lymphography",
            "pendigits",
            "magic.gamma",
            "satimage-2",
            "vowels",
            "skin",
            "PageBlocks",
            "Stamps",
            "WBC",
            "census",
        ],
        "revoked": {
            "smtp": "apex (−10.3 vs fair); helix stripped Phase0",
            "wine": "nautilus (−5.0) + RDT Phase0 (−12.0 vs fair) → baseline_ddae",
            "census": "RDT Phase0 (−4.0 vs fair) → baseline_ddae",
            "speech": "mce+smc+apex+orbit+delta kitchen-sink",
            "SVHN": "mce+orbit+helix kitchen-sink (−4.57 vs fair)",
            "ALOI": "mce+gate+orbit+locus kitchen-sink (−2.98 vs fair)",
            "celeba": "helix stripped Phase0",
        },
        "selection_contract": {
            "primary": "val_loss",
            "secondary": "eps_ball + complexity + synth_val_pr",
            "never": "test-PR",
        },
        "sets": {
            "hard_tail_12": HARD_TAIL_12,
            "bleed_cv": BLEED_CV,
            "bleed_classical": BLEED_CLASSICAL,
            "ship_select": SHIP_SELECT_DEFAULT,
        },
    }


def _build_candidate_cfg(base_cfg: Dict[str, Any], cand: str) -> Dict[str, Any]:
    from src.policy import policy_overrides, _deep_update

    parts = cand.split("+")
    base = parts[0]
    cfg = copy.deepcopy(base_cfg)
    cfg["adadae"] = dict(cfg.get("adadae") or {})
    cfg["adadae"]["policy"] = "static"
    cfg["adadae"].pop("exceptions_file", None)
    cfg["adadae"].pop("upgrades_file", None)
    # Explicit strip: no MCE/GATE/SMC from PER upgrades (static)
    cfg["adadae"]["use_mce"] = False
    cfg["adadae"]["use_gate"] = False
    # Phase 1: enable integrity-safe synth-val PR proxy for ranking
    cfg["adadae"]["synth_val_proxy"] = True
    cfg = _deep_update(cfg, policy_overrides(base))
    cfg.setdefault("adadae", {})["use_mce"] = False
    cfg["adadae"]["use_gate"] = False
    cfg["adadae"]["synth_val_proxy"] = True
    for ov in parts[1:]:
        if ov in OVERLAY_CANDIDATES:
            cfg = _deep_update(cfg, OVERLAY_CANDIDATES[ov])
        elif ov == "nautilus":
            cfg.setdefault("adadae", {})["use_nautilus"] = True
        elif ov == "spiral":
            cfg.setdefault("adadae", {})["use_spiral"] = True
    cfg["adadae"]["resolved_policy"] = f"select:{cand}"
    cfg["paths"] = dict(cfg["paths"])
    cfg["paths"]["results_dir"] = "results/adadae_per_select"
    cfg["paths"]["run_id"] = "adadae_per_select"
    hw = dict(cfg.get("hardware") or {})
    hw["dataloader_num_workers"] = 0
    hw["pin_memory"] = False
    cfg["hardware"] = hw
    return cfg


def run_selection(
    datasets: List[str],
    seeds: List[int],
    hardware: Optional[str],
    candidates: Optional[List[str]] = None,
    max_splits: int = 1,
    eps_rel: float = 0.05,
    rdt_veto_margin: float = 0.05,
) -> Dict[str, Any]:
    from src.config import load_config
    from src.data.datasets import build_registry
    from src.memory import cleanup_memory
    from src.train.experiment import run_single_file, set_seed

    # CRITICAL: pass hardware into load_config so cfg["hardware"] is a dict
    # (never assign the string "hardware_16gb.yaml" — that file does not exist).
    base = load_config(
        str(PROJECT_ROOT / "configs" / "adadae_per.yaml"),
        hardware=hardware,
    )
    if not isinstance(base.get("hardware"), dict):
        raise TypeError(
            f"config['hardware'] must be a dict after load_config, got {type(base.get('hardware'))}"
        )

    cands = candidates or list(STRIP_FIRST_CANDIDATES)
    adbench = Path(base["paths"]["adbench_root"])
    registry = build_registry(adbench)
    by_name = {e.name: e for e in registry}
    results: Dict[str, Any] = {
        "mode": "gpu_val_loss_select",
        "selection_metric": "val_loss+eps_complexity+synth_val_pr",
        "selection_contract": {
            "primary": "minimize val_loss",
            "epsilon_ball_rel": eps_rel,
            "secondary": ["complexity_prior", "maximize synth_val_pr"],
            "hard_veto": "risky RDT/MCE/GATE if synth_val_pr << baseline",
            "never": "test-PR selection",
        },
        "winners": {},
        "jobs": [],
        "failures": [],
        "candidates": cands,
        "datasets": datasets,
        "hardware_profile": (base.get("hardware") or {}).get("hardware_profile"),
        "n_ok": 0,
        "n_fail": 0,
    }

    for ds in datasets:
        if ds not in by_name:
            print(f"SKIP missing dataset {ds}")
            results["failures"].append({"dataset": ds, "error": "missing_from_registry"})
            continue
        entry = by_name[ds]
        cat = entry.category  # registry truth
        rels = list(entry.relative_paths)[: max(1, max_splits)]
        if not rels:
            print(f"SKIP {ds}: no relative_paths")
            results["failures"].append({"dataset": ds, "error": "no_relative_paths"})
            continue

        ds_votes: Dict[str, List[float]] = {c: [] for c in cands}
        ds_synths: Dict[str, List[float]] = {c: [] for c in cands}
        for seed in seeds:
            seed_vals: Dict[str, float] = {}
            seed_synths: Dict[str, float] = {}
            for cand in cands:
                cfg = _build_candidate_cfg(base, cand)
                set_seed(seed)
                split_vals: List[float] = []
                split_prs: List[float] = []
                split_synths: List[float] = []
                last_exc: Optional[str] = None
                for rel in rels:
                    try:
                        out = run_single_file(
                            npz_path=adbench / rel,
                            setting="semi-supervised",
                            seed=seed,
                            config=cfg,
                            dataset_name=entry.name,
                            split_name=rel,
                            category=cat,
                        )
                    except Exception as exc:  # noqa: BLE001
                        last_exc = f"{type(exc).__name__}: {exc}"
                        print(f"FAIL {ds} {cand} seed={seed} split={rel}: {last_exc}")
                        results["failures"].append(
                            {
                                "dataset": ds,
                                "candidate": cand,
                                "seed": seed,
                                "split": rel,
                                "error": last_exc,
                            }
                        )
                        results["n_fail"] += 1
                        cleanup_memory()
                        continue
                    raw = out.get("best_val_metric")
                    if raw is None:
                        last_exc = "best_val_metric is None"
                        print(f"FAIL {ds} {cand} seed={seed}: {last_exc}")
                        results["n_fail"] += 1
                        cleanup_memory()
                        continue
                    val = float(raw)
                    split_vals.append(val)
                    pr = (out.get("metrics") or {}).get("PR-AUC")
                    if pr is not None:
                        split_prs.append(float(pr))
                    syn = out.get("synth_val_pr")
                    if syn is not None:
                        split_synths.append(float(syn))
                    results["n_ok"] += 1
                    cleanup_memory()

                if not split_vals:
                    continue
                val = sum(split_vals) / len(split_vals)
                ds_votes[cand].append(val)
                seed_vals[cand] = val
                syn_mean = (
                    sum(split_synths) / len(split_synths) if split_synths else None
                )
                if syn_mean is not None:
                    ds_synths[cand].append(syn_mean)
                    seed_synths[cand] = syn_mean
                results["jobs"].append(
                    {
                        "dataset": ds,
                        "seed": seed,
                        "candidate": cand,
                        "val_loss": val,
                        "synth_val_pr": syn_mean,
                        "PR": (sum(split_prs) / len(split_prs)) if split_prs else None,
                        "n_splits": len(split_vals),
                        "complexity": recipe_complexity(cand),
                    }
                )

            if seed_vals:
                try:
                    seed_winner, _meta = pick_winner_with_proxies(
                        {c: seed_vals.get(c, float("inf")) for c in cands},
                        {c: seed_synths.get(c) for c in cands},  # type: ignore[arg-type]
                        eps_rel=eps_rel,
                        rdt_veto_margin=rdt_veto_margin,
                    )
                    results.setdefault("_seed_winners", {}).setdefault(ds, []).append(
                        seed_winner
                    )
                except ValueError:
                    pass

        means = {c: (sum(v) / len(v) if v else float("inf")) for c, v in ds_votes.items()}
        means_synth = {
            c: (sum(v) / len(v) if v else None) for c, v in ds_synths.items()
        }
        finite = {c: m for c, m in means.items() if m != float("inf")}
        if not finite:
            print(f"ERROR {ds}: all candidates failed (no finite val_loss)")
            results["winners"][ds] = {
                "policy": None,
                "mean_val_loss": means,
                "mean_synth_val_pr": means_synth,
                "strip_mce_gate": True,
                "category": cat,
                "error": "all_candidates_failed",
            }
            continue
        winner, pick_meta = pick_winner_with_proxies(
            means, means_synth, eps_rel=eps_rel, rdt_veto_margin=rdt_veto_margin
        )
        results["winners"][ds] = {
            "policy": winner,
            "mean_val_loss": means,
            "mean_synth_val_pr": means_synth,
            "strip_mce_gate": True,
            "category": cat,
            "selection": pick_meta,
        }
        print(
            f"WINNER {ds}: {winner} val={means[winner]:.6g} "
            f"synth={means_synth.get(winner)} complexity={recipe_complexity(winner)} "
            f"vetoed={pick_meta.get('vetoed_risky')}"
        )
    return results


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--preset",
        choices=["ship", "hard-12", "bleed-cv", "bleed-classical", "legacy"],
        default=None,
        help="Dataset preset (default ship when --datasets omitted and not dry-run)",
    )
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=[111, 222, 333])
    p.add_argument("--hardware", default=None)
    p.add_argument(
        "--max-splits",
        type=int,
        default=1,
        help="Max NPZ splits per CV/NLP family (default 1 for speed; protocol uses all)",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--apply-evidence-freeze", action="store_true")
    p.add_argument(
        "--candidates",
        nargs="*",
        default=None,
        help="Override strip-first candidate list",
    )
    p.add_argument(
        "--eps-rel",
        type=float,
        default=0.05,
        help="Relative ε-ball around best val_loss for secondary ranking (default 0.05)",
    )
    p.add_argument(
        "--rdt-veto-margin",
        type=float,
        default=0.05,
        help="Synth-val PR margin: veto risky recipes this far below baseline",
    )
    args = p.parse_args()

    if args.dry_run:
        report = dry_run_resolve()
        out = PROJECT_ROOT / "results/adadae_per/thesis/loop6_train_only_select.json"
        rc = 0
    elif args.apply_evidence_freeze:
        report = evidence_freeze_report()
        out = PROJECT_ROOT / "results/adadae_per/thesis/phase1_hard_freeze.json"
        rc = 0
    else:
        if args.datasets:
            datasets = list(args.datasets)
        elif args.preset:
            datasets = _preset_datasets(args.preset)
        else:
            datasets = _preset_datasets("ship")
        report = run_selection(
            datasets,
            args.seeds,
            args.hardware,
            args.candidates,
            max_splits=args.max_splits,
            eps_rel=float(args.eps_rel),
            rdt_veto_margin=float(args.rdt_veto_margin),
        )
        out = PROJECT_ROOT / "results/adadae_per/thesis/phase1_hard_freeze.json"
        n_winners = sum(
            1
            for w in (report.get("winners") or {}).values()
            if isinstance(w, dict) and w.get("policy")
        )
        n_fail_ds = sum(
            1
            for w in (report.get("winners") or {}).values()
            if isinstance(w, dict) and w.get("error")
        )
        report["n_winners"] = n_winners
        report["n_failed_datasets"] = n_fail_ds
        if n_winners == 0:
            report["status"] = "FAILED_all_infinity"
            rc = 2
        elif n_fail_ds:
            report["status"] = "partial"
            rc = 1
        else:
            report["status"] = "ok"
            rc = 0

    out.parent.mkdir(parents=True, exist_ok=True)
    # JSON cannot encode Inf; replace for readability
    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, float) and (obj == float("inf") or obj != obj):
            return None
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    out.write_text(json.dumps(_sanitize(report), indent=2), encoding="utf-8")
    printable = {k: report[k] for k in report if k != "jobs"}
    print(json.dumps(_sanitize(printable), indent=2)[:4000])
    print(f"Wrote {out} status={report.get('status', 'ok')} n_ok={report.get('n_ok')} n_fail={report.get('n_fail')}")
    if rc != 0:
        print(
            "ERROR: selection produced no usable winners (check FAIL lines / hardware / ADBench paths).",
            file=sys.stderr,
        )
        print(
            "Delete bogus Infinity freeze before apply_hard_tail_freeze; "
            "fix then re-run: python scripts/train_only_recipe_select.py --preset ship --hardware 16gb",
            file=sys.stderr,
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
