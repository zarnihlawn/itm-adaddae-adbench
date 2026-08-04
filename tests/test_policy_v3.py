"""Tests for AdaDDAE v3 policy routing."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.policy import (
    apply_routed_config,
    load_policy_exceptions,
    resolve_paradigm_policy_name,
    resolve_policy_name,
)


@pytest.fixture
def routed_config():
    return {
        "adadae": {"policy": "routed"},
        "train": {"epochs": 100},
        "diffusion": {"num_timesteps": 50},
    }


def test_unsup_default_is_ssts():
    assert resolve_policy_name("unsupervised", "classical", "cardio") == "unsup_ssts"


def test_unsup_vowels_fallback():
    assert resolve_policy_name("unsupervised", "classical", "vowels") == "unsup_baseline_fallback"


def test_semi_classical_baseline():
    assert resolve_policy_name("semi-supervised", "classical", "cardio") == "baseline_ddae"


def test_semi_nlp_baseline():
    assert resolve_policy_name("semi-supervised", "nlp", "Agnews") == "semi_nlp_baseline"


def test_semi_cv_ftp():
    assert resolve_policy_name("semi-supervised", "cv", "CIFAR10") == "semi_cvnlp_ftp"


def test_semi_speech_specialist():
    assert resolve_policy_name("semi-supervised", "classical", "speech") == "semi_speech_specialist"


def test_apply_routed_sets_resolved_policy(routed_config):
    out = apply_routed_config(
        routed_config, "unsupervised", "classical", "vowels", meta={"n": 1000, "d": 12}
    )
    assert out["adadae"]["resolved_policy"] == "unsup_baseline_fallback"
    assert out["adadae"]["use_danc"] is False


def test_exceptions_file_loads():
    exc = load_policy_exceptions()
    assert "vowels" in exc["unsup_baseline_fallback"]
    assert exc["semi_specialists"]["speech"] == "semi_speech_specialist"


def test_paradigm_setting_only():
    assert resolve_paradigm_policy_name("unsupervised") == "unsup_ssts"
    assert resolve_paradigm_policy_name("semi-supervised") == "champion_semi"
    cfg = {
        "adadae": {
            "policy": "paradigm",
            "use_uncertainty_view": False,
            "fusion_mode": "fixed",
        },
        "train": {"contrastive": False},
        "features": {"scaler": "standard"},
    }
    u = apply_routed_config(cfg, "unsupervised", "classical", "speech")
    s = apply_routed_config(cfg, "semi-supervised", "classical", "speech")
    assert u["adadae"]["resolved_policy"] == "paradigm_unsup_ssts"
    assert s["adadae"]["resolved_policy"] == "paradigm_champion_semi"
    # Dataset name must not change paradigm branch (no specialists)
    assert u["adadae"]["use_danc"] is True
    assert s["adadae"]["use_danc"] is False
    assert s["adadae"]["use_ftp"] is True
    assert s["adadae"]["use_uncertainty_view"] is False
