#!/usr/bin/env python3
"""Unit tests for train-only recipe select veto + composed upgrades."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_only_recipe_select import (  # noqa: E402
    _build_candidate_cfg,
    pick_winner_with_proxies,
)


def test_rdt_vetoed_on_large_val_loss_ratio() -> None:
    # FashionMNIST-like: RDT val_loss ~3.4× lower, synth not a clear unsaturated win
    means_val = {
        "baseline_ddae": 2.0747,
        "semi_rdt_tail": 0.6173,
        "champion_semi": 3.08,
    }
    means_synth = {
        "baseline_ddae": 0.9749,
        "semi_rdt_tail": 0.9930,
        "champion_semi": 0.9760,
    }
    winner, meta = pick_winner_with_proxies(
        means_val, means_synth, max_val_loss_ratio=2.5
    )
    assert winner == "baseline_ddae", winner
    assert "semi_rdt_tail" in meta["vetoed_risky"]
    assert meta["veto_reasons"]["semi_rdt_tail"].startswith("val_loss_ratio")


def test_hepatitis_like_rdt_survives_ratio_under_cap() -> None:
    means_val = {
        "baseline_ddae": 0.4219,
        "semi_rdt_tail": 0.1953,
    }
    means_synth = {
        "baseline_ddae": 1.0,
        "semi_rdt_tail": 1.0,
    }
    winner, meta = pick_winner_with_proxies(
        means_val, means_synth, max_val_loss_ratio=2.5
    )
    # ratio ≈ 2.16 < 2.5 and synth saturated → RDT may win ε-ball
    assert winner == "semi_rdt_tail", (winner, meta)
    assert "semi_rdt_tail" not in meta["vetoed_risky"]


def test_internetads_extreme_ratio_vetoed() -> None:
    means_val = {"baseline_ddae": 0.2416, "semi_rdt_tail": 0.0052}
    means_synth = {"baseline_ddae": 1.0, "semi_rdt_tail": 1.0}
    winner, meta = pick_winner_with_proxies(means_val, means_synth)
    assert winner == "baseline_ddae"
    assert "semi_rdt_tail" in meta["vetoed_risky"]


def test_composed_upgrades_apply_taps_for_listed_dataset() -> None:
    from src.config import load_config

    base = load_config(str(PROJECT_ROOT / "configs" / "adadae_per.yaml"), hardware="cpu")
    # Imdb is the only remaining contrastive_taps_semi member after last-shot
    cfg = _build_candidate_cfg(
        base, "baseline_ddae", dataset_name="Imdb", category="nlp"
    )
    assert cfg["train"].get("contrastive") is True
    assert "taps" in str(cfg["adadae"].get("resolved_policy"))


def test_composed_upgrades_no_taps_on_amazon_after_strip() -> None:
    from src.config import load_config
    from src.policy_per import clear_per_upgrades_cache

    clear_per_upgrades_cache()
    base = load_config(str(PROJECT_ROOT / "configs" / "adadae_per.yaml"), hardware="cpu")
    cfg = _build_candidate_cfg(
        base, "baseline_ddae", dataset_name="Amazon", category="nlp"
    )
    assert not cfg.get("train", {}).get("contrastive")
    assert "taps" not in str(cfg["adadae"].get("resolved_policy"))


def test_composed_upgrades_no_taps_on_backdoor_after_strip() -> None:
    from src.config import load_config
    from src.policy_per import clear_per_upgrades_cache

    clear_per_upgrades_cache()
    base = load_config(str(PROJECT_ROOT / "configs" / "adadae_per.yaml"), hardware="cpu")
    cfg = _build_candidate_cfg(
        base, "baseline_ddae", dataset_name="backdoor", category="classical"
    )
    assert not cfg.get("train", {}).get("contrastive")
    pol = str(cfg["adadae"].get("resolved_policy"))
    assert "taps" not in pol
    assert "cal_fuse" not in pol


def test_phase0_locks_include_full57_disasters() -> None:
    from scripts.apply_hard_tail_freeze import PHASE0_LOCKS

    for ds in (
        "wine",
        "census",
        "FashionMNIST",
        "MNIST-C",
        "InternetAds",
        "optdigits",
    ):
        assert PHASE0_LOCKS[ds] == "baseline_ddae"


if __name__ == "__main__":
    test_rdt_vetoed_on_large_val_loss_ratio()
    test_hepatitis_like_rdt_survives_ratio_under_cap()
    test_internetads_extreme_ratio_vetoed()
    test_composed_upgrades_apply_taps_for_listed_dataset()
    test_composed_upgrades_no_taps_on_amazon_after_strip()
    test_composed_upgrades_no_taps_on_backdoor_after_strip()
    test_phase0_locks_include_full57_disasters()
    print("ok")
