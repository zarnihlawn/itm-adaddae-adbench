#!/usr/bin/env python3
"""
Ablation ladder for thesis (cumulative component isolation).

Steps: ddae_repro -> adadae_fixed -> ftp -> lfdanc -> ssts -> taps -> vus -> full_adadae
       oracle_danc (upper-bound ablation)
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.datasets import build_registry
from src.eval.metrics import mean_std_metrics
from src.memory import cleanup_memory
from src.runlog.logger import RunLogger
from src.train.experiment import run_single_file


ABLATIONS = {
    "ddae_repro": {
        "adadae": {
            "use_danc": False,
            "use_scs": False,
            "use_ftp": False,
            "use_multiview": False,
            "scs_max_timesteps": 50,
            "scs_mode": "full_sum",
            "scs_selection": "linspace",
            "contrastive_pairing": "random",
            "use_uncertainty_view": False,
            "fusion_mode": "fixed",
            "fusion_weights": {"reconstruction": 1.0, "latent": 0.0, "residual": 0.0, "uncertainty": 0.0},
        },
        "train": {"contrastive": False, "contrastive_alpha": 0.0},
        "features": {"scaler": "standard", "pca_dim_threshold": 99999, "clip_outliers": False},
        "diffusion": {"num_timesteps": 50, "scheduler": "linear", "time_emb_dim": 4},
    },
    "adadae_fixed": {
        "adadae": {
            "use_danc": False,
            "use_scs": True,
            "use_ftp": False,
            "use_multiview": True,
            "scs_mode": "full_sum",
            "scs_selection": "linspace",
            "scs_max_timesteps": 50,
            "contrastive_pairing": "random",
            "use_uncertainty_view": False,
            "fusion_mode": "fixed",
            "fusion_weights": {"reconstruction": 0.6, "latent": 0.3, "residual": 0.1, "uncertainty": 0.0},
        },
        "train": {"contrastive": False, "contrastive_alpha": 0.0},
        "features": {"scaler": "standard", "pca_dim_threshold": 99999, "clip_outliers": False},
        "diffusion": {"num_timesteps": 50, "scheduler": "linear", "time_emb_dim": 4},
    },
    "ftp": {
        "adadae": {
            "use_danc": False,
            "use_scs": False,
            "use_ftp": True,
            "use_multiview": False,
            "use_uncertainty_view": False,
            "contrastive_pairing": "random",
        },
        "train": {"contrastive": False, "contrastive_alpha": 0.0},
    },
    "lfdanc": {
        "adadae": {
            "use_danc": True,
            "use_scs": False,
            "use_ftp": True,
            "use_multiview": False,
            "danc_contamination_mode": "label_free",
            "use_uncertainty_view": False,
            "contrastive_pairing": "random",
        },
        "train": {"contrastive": False, "contrastive_alpha": 0.0},
    },
    "ssts": {
        "adadae": {
            "use_danc": True,
            "use_scs": True,
            "use_ftp": True,
            "use_multiview": False,
            "scs_mode": "snr_weighted",
            "scs_selection": "snr_stratified",
            "danc_contamination_mode": "label_free",
            "use_uncertainty_view": False,
            "use_dte_view": False,
            "use_rejection_training": False,
            "contrastive_pairing": "random",
        },
        "train": {"contrastive": False, "contrastive_alpha": 0.0},
    },
    "rdt": {
        "adadae": {
            "use_danc": True,
            "use_scs": True,
            "use_ftp": True,
            "use_multiview": False,
            "scs_mode": "snr_weighted",
            "scs_selection": "snr_stratified",
            "danc_contamination_mode": "label_free",
            "use_rejection_training": True,
            "rejection_quantile": 0.95,
            "use_dte_view": False,
            "use_uncertainty_view": False,
            "contrastive_pairing": "random",
        },
        "train": {"contrastive": False, "contrastive_alpha": 0.0},
    },
    "taps": {
        "adadae": {
            "use_danc": True,
            "use_scs": True,
            "use_ftp": True,
            "use_multiview": False,
            "scs_mode": "snr_weighted",
            "scs_selection": "snr_stratified",
            "danc_contamination_mode": "label_free",
            "contrastive_pairing": "taps",
            "use_uncertainty_view": False,
        },
        "train": {"contrastive": True, "contrastive_alpha": 0.15, "hard_negative_mining": True},
    },
    "vus": {
        "adadae": {
            "use_danc": True,
            "use_scs": True,
            "use_ftp": True,
            "use_multiview": True,
            "scs_mode": "snr_weighted",
            "scs_selection": "snr_stratified",
            "danc_contamination_mode": "label_free",
            "contrastive_pairing": "taps",
            "use_rejection_training": True,
            "use_uncertainty_view": True,
            "use_dte_view": False,
            "uncertainty_draws": 3,
            "fusion_mode": "calibrated",
            "fusion_weights": {
                "reconstruction": 0.45,
                "latent": 0.2,
                "residual": 0.1,
                "uncertainty": 0.15,
                "diffusion_time": 0.0,
            },
        },
        "train": {"contrastive": True, "contrastive_alpha": 0.15, "hard_negative_mining": True},
    },
    "dte": {
        "adadae": {
            "use_danc": True,
            "use_scs": True,
            "use_ftp": True,
            "use_multiview": True,
            "scs_mode": "snr_weighted",
            "scs_selection": "snr_stratified",
            "danc_contamination_mode": "label_free",
            "contrastive_pairing": "taps",
            "use_rejection_training": True,
            "use_uncertainty_view": True,
            "use_dte_view": True,
            "dte_knn": 5,
            "fusion_mode": "calibrated",
            "fusion_weights": {
                "reconstruction": 0.4,
                "latent": 0.2,
                "residual": 0.1,
                "uncertainty": 0.1,
                "diffusion_time": 0.2,
            },
        },
        "train": {"contrastive": True, "contrastive_alpha": 0.15, "hard_negative_mining": True},
    },
    "full_adadae": {
        "adadae": {
            "use_danc": True,
            "use_scs": True,
            "use_ftp": True,
            "use_multiview": True,
            "scs_max_timesteps": 32,
            "scs_mode": "snr_weighted",
            "scs_selection": "snr_stratified",
            "danc_contamination_mode": "label_free",
            "contrastive_pairing": "taps",
            "use_rejection_training": True,
            "use_uncertainty_view": True,
            "use_dte_view": True,
            "uncertainty_draws": 3,
            "dte_knn": 5,
            "fusion_mode": "calibrated",
            "fusion_weights": {
                "reconstruction": 0.4,
                "latent": 0.2,
                "residual": 0.1,
                "uncertainty": 0.1,
                "diffusion_time": 0.2,
            },
        },
        "train": {"contrastive": True, "contrastive_alpha": 0.15, "hard_negative_mining": True},
    },
    "oracle_danc": {
        "adadae": {
            "use_danc": True,
            "use_scs": True,
            "use_ftp": True,
            "use_multiview": True,
            "scs_mode": "snr_weighted",
            "scs_selection": "snr_stratified",
            "danc_contamination_mode": "oracle",
            "contrastive_pairing": "taps",
            "use_uncertainty_view": True,
            "fusion_mode": "calibrated",
        },
        "train": {"contrastive": True, "contrastive_alpha": 0.15},
    },
    # AdaDDAE-2 ladder
    "adadae2_core": {
        "adadae": {
            "use_danc": True,
            "use_chronos": False,
            "use_scs": True,
            "use_ftp": True,
            "use_multiview": True,
            "use_uncertainty_view": False,
            "use_dte_view": True,
            "use_geode": False,
            "use_aether": False,
            "use_nexus": False,
            "fusion_mode": "calibrated",
            "use_rejection_training": True,
            "contrastive_pairing": "taps",
        },
        "train": {"contrastive": True, "contrastive_alpha": 0.15},
    },
    "chronos": {
        "adadae": {
            "use_danc": True,
            "use_chronos": True,
            "use_scs": True,
            "use_ftp": True,
            "use_multiview": True,
            "use_uncertainty_view": False,
            "use_dte_view": True,
            "use_geode": False,
            "use_aether": False,
            "use_nexus": False,
            "fusion_mode": "calibrated",
            "contrastive_pairing": "taps",
        },
        "train": {"contrastive": True, "contrastive_alpha": 0.15},
    },
    "geode": {
        "adadae": {
            "use_danc": True,
            "use_chronos": True,
            "use_scs": True,
            "use_ftp": True,
            "use_multiview": True,
            "use_uncertainty_view": False,
            "use_dte_view": True,
            "use_geode": True,
            "use_aether": False,
            "use_nexus": False,
            "fusion_mode": "calibrated",
            "contrastive_pairing": "taps",
        },
        "train": {"contrastive": True, "contrastive_alpha": 0.15},
    },
    "calix": {
        "adadae": {
            "use_danc": True,
            "use_chronos": True,
            "use_scs": True,
            "use_ftp": True,
            "use_multiview": True,
            "use_uncertainty_view": False,
            "use_dte_view": True,
            "use_geode": True,
            "use_aether": False,
            "use_nexus": False,
            "fusion_mode": "calix",
            "contrastive_pairing": "taps",
        },
        "train": {"contrastive": True, "contrastive_alpha": 0.15},
    },
    "nexus": {
        "adadae": {
            "use_danc": True,
            "use_chronos": True,
            "use_scs": True,
            "use_ftp": True,
            "use_multiview": True,
            "use_uncertainty_view": False,
            "use_dte_view": True,
            "use_geode": True,
            "use_aether": False,
            "use_nexus": True,
            "fusion_mode": "calix",
            "contrastive_pairing": "taps",
        },
        "train": {"contrastive": True, "contrastive_alpha": 0.15},
    },
    "aether": {
        "adadae": {
            "use_danc": True,
            "use_chronos": True,
            "use_scs": True,
            "use_ftp": True,
            "use_multiview": True,
            "use_uncertainty_view": False,
            "use_dte_view": True,
            "use_geode": True,
            "use_aether": True,
            "use_nexus": True,
            "fusion_mode": "calix",
            "contrastive_pairing": "taps",
        },
        "train": {"contrastive": True, "contrastive_alpha": 0.15},
    },
    "full_adadae2": {
        "adadae": {
            "use_danc": True,
            "use_chronos": True,
            "use_scs": True,
            "use_ftp": True,
            "use_multiview": True,
            "use_uncertainty_view": False,
            "use_dte_view": True,
            "use_geode": True,
            "use_aether": True,
            "use_nexus": True,
            "fusion_mode": "calix",
            "use_mce": False,
            "use_gate": False,
            "contrastive_pairing": "taps",
            "use_rejection_training": True,
        },
        "train": {"contrastive": True, "contrastive_alpha": 0.15},
    },
    # AdaDDAE-3 ladder
    "adadae3_core": {
        "adadae": {
            "use_danc": True,
            "use_chronos": True,
            "use_scs": True,
            "use_ftp": True,
            "use_multiview": True,
            "use_dte_view": True,
            "use_geode": True,
            "use_aether": True,
            "use_helios": False,
            "use_kairos": False,
            "use_orbis": False,
            "use_plexus": False,
            "use_mirage": False,
            "use_nexus_v2": False,
            "use_rdt_v2": False,
            "use_atlas": False,
            "use_flux": False,
            "use_aegis": False,
            "fusion_mode": "calix",
            "use_mce": False,
            "use_gate": False,
        },
        "train": {"contrastive": True, "contrastive_alpha": 0.15},
    },
    "helios_kairos": {
        "adadae": {
            "use_chronos": True,
            "use_geode": True,
            "use_aether": True,
            "use_helios": True,
            "use_kairos": True,
            "fusion_mode": "calix",
            "use_mce": False,
            "use_gate": False,
        },
        "train": {"contrastive": True, "contrastive_alpha": 0.15},
    },
    "orbis_plexus": {
        "adadae": {
            "use_chronos": True,
            "use_geode": True,
            "use_aether": True,
            "use_helios": True,
            "use_kairos": True,
            "use_orbis": True,
            "use_plexus": True,
            "fusion_mode": "calix",
            "use_mce": False,
            "use_gate": False,
        },
        "train": {"contrastive": True, "contrastive_alpha": 0.15},
    },
    "argos_aegis": {
        "adadae": {
            "use_chronos": True,
            "use_geode": True,
            "use_aether": True,
            "use_helios": True,
            "use_kairos": True,
            "use_orbis": True,
            "use_plexus": True,
            "use_argos": True,
            "use_aegis": True,
            "fusion_mode": "argos",
            "use_mce": False,
            "use_gate": False,
        },
        "train": {"contrastive": True, "contrastive_alpha": 0.15},
    },
    "mirage_nexusv2": {
        "adadae": {
            "use_chronos": True,
            "use_geode": True,
            "use_aether": True,
            "use_helios": True,
            "use_kairos": True,
            "use_orbis": True,
            "use_plexus": True,
            "use_argos": True,
            "use_aegis": True,
            "use_mirage": True,
            "use_nexus_v2": True,
            "use_rdt_v2": True,
            "fusion_mode": "argos",
            "use_mce": False,
            "use_gate": False,
        },
        "train": {"contrastive": True, "contrastive_alpha": 0.15},
    },
    "atlas_flux": {
        "adadae": {
            "use_chronos": True,
            "use_geode": True,
            "use_aether": True,
            "use_helios": True,
            "use_kairos": True,
            "use_orbis": True,
            "use_plexus": True,
            "use_argos": True,
            "use_aegis": True,
            "use_mirage": True,
            "use_nexus_v2": True,
            "use_rdt_v2": True,
            "use_atlas": True,
            "use_flux": True,
            "use_epoche": True,
            "fusion_mode": "argos",
            "use_mce": False,
            "use_gate": False,
        },
        "train": {"contrastive": True, "contrastive_alpha": 0.15},
    },
    "full_adadae3": {
        "adadae": {
            "use_danc": True,
            "use_chronos": True,
            "use_scs": True,
            "use_ftp": True,
            "use_multiview": True,
            "use_dte_view": True,
            "use_geode": True,
            "use_aether": True,
            "use_nexus": False,
            "use_helios": True,
            "use_kairos": True,
            "use_orbis": True,
            "use_plexus": True,
            "use_argos": False,
            "use_aegis": False,
            "use_mirage": False,
            "use_nexus_v2": True,
            "use_rdt_v2": True,
            "use_atlas": False,
            "use_flux": False,
            "use_epoche": True,
            "fusion_mode": "calix",
            "use_uncertainty_view": False,
            "use_mce": False,
            "use_gate": False,
            "use_rejection_training": True,
            "contrastive_pairing": "taps",
        },
        "train": {"contrastive": True, "contrastive_alpha": 0.15},
    },
    "adadae4_core": {
        "adadae": {
            "use_chronos": True,
            "use_geode": True,
            "use_aether": True,
            "use_helios": True,
            "use_kairos": True,
            "use_omni": False,
            "auto_regime_gates": False,
            "use_nano": False,
            "use_torrent": False,
            "use_prism": False,
            "use_polis": False,
            "use_sieve": False,
            "use_quell": False,
            "fusion_mode": "calix",
            "use_mce": False,
            "use_gate": False,
        },
        "train": {"contrastive": True, "contrastive_alpha": 0.15},
    },
    "omni_gates": {
        "adadae": {
            "use_chronos": True,
            "use_geode": True,
            "use_aether": True,
            "use_helios": True,
            "use_omni": True,
            "auto_regime_gates": True,
            "use_nano": True,
            "use_torrent": True,
            "use_prism": True,
            "use_polis": True,
            "use_sieve": True,
            "use_needle": True,
            "use_robust": True,
            "use_quell": True,
            "fusion_mode": "quell",
            "use_mce": False,
            "use_gate": False,
        },
        "train": {"contrastive": True, "contrastive_alpha": 0.15},
    },
    "full_adadae4": {
        "adadae": {
            "use_danc": True,
            "use_chronos": True,
            "use_scs": True,
            "use_ftp": True,
            "use_multiview": True,
            "use_dte_view": True,
            "use_geode": True,
            "use_aether": True,
            "use_helios": True,
            "use_kairos": True,
            "use_orbis": True,
            "use_plexus": True,
            "use_nexus_v2": True,
            "use_rdt_v2": True,
            "use_epoche": True,
            "use_mirage": False,
            "use_flux": False,
            "use_atlas": False,
            "use_omni": True,
            "auto_regime_gates": True,
            "use_nano": True,
            "use_torrent": True,
            "use_prism": True,
            "use_polis": True,
            "use_sieve": True,
            "use_needle": True,
            "use_sparse_view": True,
            "use_robust": True,
            "use_quell": True,
            "fusion_mode": "quell",
            "use_mce": False,
            "use_gate": False,
            "contrastive_pairing": "taps",
            "use_rejection_training": True,
        },
        "train": {"contrastive": True, "contrastive_alpha": 0.15},
    },
    # AdaDDAE-5 ladder (A1 core + A5 modules)
    "adadae5_core": {
        "adadae": {
            "use_danc": True,
            "use_scs": True,
            "use_ftp": True,
            "use_multiview": True,
            "use_dte_view": True,
            "use_uncertainty_view": True,
            "use_rejection_training": True,
            "fusion_mode": "calibrated",
            "use_figaro": False,
            "use_dsm_plus": False,
            "use_mahala": False,
            "use_full_dte": False,
            "use_lexicon": False,
            "use_pura": False,
            "use_evt_tail": False,
            "use_confal": False,
            "use_spectra": False,
            "use_sinkhorn": False,
            "use_ib_latent": False,
            "use_elbo_s": False,
            "use_curriculum_snr": False,
            "use_vmf_z": False,
            "use_mce": False,
            "use_gate": False,
            "contrastive_pairing": "taps",
        },
        "train": {"contrastive": True, "contrastive_alpha": 0.15},
    },
    "figaro_dsm": {
        "adadae": {
            "use_danc": True,
            "use_scs": True,
            "use_ftp": True,
            "use_multiview": True,
            "use_dte_view": True,
            "fusion_mode": "calibrated",
            "use_figaro": True,
            "use_dsm_plus": True,
            "use_mahala": False,
            "use_full_dte": False,
            "use_lexicon": False,
            "use_mce": False,
            "use_gate": False,
        },
        "train": {"contrastive": True, "contrastive_alpha": 0.15},
    },
    "mahala_dte": {
        "adadae": {
            "use_danc": True,
            "use_scs": True,
            "use_ftp": True,
            "use_multiview": True,
            "use_dte_view": True,
            "fusion_mode": "calibrated",
            "use_figaro": True,
            "use_dsm_plus": True,
            "use_mahala": True,
            "use_full_dte": True,
            "use_lexicon": False,
            "use_mce": False,
            "use_gate": False,
        },
        "train": {"contrastive": True, "contrastive_alpha": 0.15},
    },
    "lexicon_fuse": {
        "adadae": {
            "use_danc": True,
            "use_scs": True,
            "use_ftp": True,
            "use_multiview": True,
            "use_dte_view": True,
            "use_figaro": True,
            "use_dsm_plus": True,
            "use_mahala": True,
            "use_full_dte": True,
            "use_lexicon": True,
            "fusion_mode": "lexicon",
            "use_mce": False,
            "use_gate": False,
        },
        "train": {"contrastive": True, "contrastive_alpha": 0.15},
    },
    "full_adadae5": {
        "adadae": {
            "use_danc": True,
            "use_scs": True,
            "use_ftp": True,
            "use_multiview": True,
            "use_dte_view": True,
            "use_uncertainty_view": True,
            "use_rejection_training": True,
            "use_figaro": True,
            "use_dsm_plus": True,
            "use_mahala": True,
            "use_full_dte": True,
            "use_lexicon": True,
            "fusion_mode": "lexicon",
            "use_pura": True,
            "use_evt_tail": True,
            "use_confal": True,
            "use_spectra": True,
            "use_sinkhorn": True,
            "use_ib_latent": True,
            "use_elbo_s": True,
            "use_curriculum_snr": True,
            "use_vmf_z": True,
            "use_mce": False,
            "use_gate": False,
            "contrastive_pairing": "taps",
        },
        "train": {"contrastive": True, "contrastive_alpha": 0.15},
    },
    # AdaDDAE-6 ladder
    "adadae6_core": {
        "adadae": {
            "use_danc": True,
            "use_scs": True,
            "use_ftp": True,
            "use_multiview": True,
            "use_dte_view": True,
            "use_uncertainty_view": True,
            "use_rejection_training": True,
            "fusion_mode": "lexicon",
            "use_figaro": True,
            "use_dsm_plus": True,
            "use_mahala": True,
            "use_full_dte": True,
            "use_lexicon": True,
            "use_helix": False,
            "use_delta": False,
            "use_apex": False,
            "use_nautilus": False,
            "use_torque": False,
            "use_orbit": False,
            "use_kale": False,
            "use_ridge": False,
            "use_locus": False,
            "use_spiral": False,
            "use_mce": False,
            "use_gate": False,
        },
        "train": {"contrastive": True, "contrastive_alpha": 0.15},
    },
    "helix_delta": {
        "adadae": {
            "use_figaro": True,
            "use_helix": True,
            "use_delta": True,
            "use_apex": False,
            "use_kale": False,
            "fusion_mode": "lexicon",
            "use_mce": False,
            "use_gate": False,
        },
        "train": {"contrastive": True, "contrastive_alpha": 0.15},
    },
    "apex_calib": {
        "adadae": {
            "use_helix": True,
            "use_delta": True,
            "use_apex": True,
            "use_evt_tail": True,
            "use_confal": True,
            "use_kale": False,
            "fusion_mode": "lexicon",
            "use_mce": False,
            "use_gate": False,
        },
        "train": {"contrastive": True, "contrastive_alpha": 0.15},
    },
    "kale_fuse": {
        "adadae": {
            "use_helix": True,
            "use_delta": True,
            "use_apex": True,
            "use_orbit": True,
            "use_locus": True,
            "use_spiral": True,
            "use_kale": True,
            "fusion_mode": "kale",
            "use_mce": False,
            "use_gate": False,
        },
        "train": {"contrastive": True, "contrastive_alpha": 0.15},
    },
    "full_adadae6": {
        "adadae": {
            "use_danc": True,
            "use_scs": True,
            "use_ftp": True,
            "use_multiview": True,
            "use_dte_view": True,
            "use_uncertainty_view": True,
            "use_rejection_training": True,
            "use_figaro": True,
            "use_dsm_plus": True,
            "use_mahala": True,
            "use_full_dte": True,
            "use_lexicon": False,
            "fusion_mode": "kale",
            "use_pura": True,
            "use_evt_tail": True,
            "use_confal": True,
            "use_spectra": True,
            "use_sinkhorn": True,
            "use_ib_latent": True,
            "use_elbo_s": False,
            "use_curriculum_snr": True,
            "use_vmf_z": True,
            "use_helix": True,
            "use_delta": True,
            "use_apex": True,
            "use_nautilus": True,
            "use_torque": True,
            "use_orbit": True,
            "use_kale": True,
            "use_ridge": True,
            "use_locus": True,
            "use_spiral": True,
            "use_mce": False,
            "use_gate": False,
            "contrastive_pairing": "taps",
        },
        "train": {"contrastive": True, "contrastive_alpha": 0.15},
    },
    # Legacy aliases
    "danc": {
        "adadae": {"use_danc": True, "use_scs": False, "use_ftp": True, "use_multiview": False},
        "train": {"contrastive": False, "contrastive_alpha": 0.0},
    },
    "scs": {
        "adadae": {"use_danc": True, "use_scs": True, "use_ftp": True, "use_multiview": False},
        "train": {"contrastive": False, "contrastive_alpha": 0.0},
    },
    "contrastive": {
        "adadae": {"use_danc": True, "use_scs": True, "use_ftp": True, "use_multiview": False},
        "train": {"contrastive": True, "contrastive_alpha": 0.2, "hard_negative_mining": True},
    },
}

LADDER_ORDER = [
    "ddae_repro",
    "adadae_fixed",
    "ftp",
    "lfdanc",
    "ssts",
    "rdt",
    "taps",
    "vus",
    "dte",
    "full_adadae",
    "oracle_danc",
    "adadae2_core",
    "chronos",
    "geode",
    "calix",
    "nexus",
    "aether",
    "full_adadae2",
    "adadae3_core",
    "helios_kairos",
    "orbis_plexus",
    "argos_aegis",
    "mirage_nexusv2",
    "atlas_flux",
    "full_adadae3",
    "adadae4_core",
    "omni_gates",
    "full_adadae4",
    "adadae5_core",
    "figaro_dsm",
    "mahala_dte",
    "lexicon_fuse",
    "full_adadae5",
    "adadae6_core",
    "helix_delta",
    "apex_calib",
    "kale_fuse",
    "full_adadae6",
]

# Leave-one-component-out from full_adadae2
LEAVE_ONE_OUT = {
    "loo_no_chronos": {"adadae": {"use_chronos": False}},
    "loo_no_geode": {"adadae": {"use_geode": False}},
    "loo_no_calix": {"adadae": {"fusion_mode": "calibrated"}},
    "loo_no_nexus": {"adadae": {"use_nexus": False}},
    "loo_no_aether": {"adadae": {"use_aether": False}},
    "loo_no_dte": {"adadae": {"use_dte_view": False}},
}

LEAVE_ONE_OUT_A3 = {
    "loo_no_helios": {"adadae": {"use_helios": False}},
    "loo_no_kairos": {"adadae": {"use_kairos": False}},
    "loo_no_orbis": {"adadae": {"use_orbis": False}},
    "loo_no_plexus": {"adadae": {"use_plexus": False}},
    "loo_no_geode": {"adadae": {"use_geode": False}},
    "loo_no_nexus_v2": {"adadae": {"use_nexus_v2": False}},
    "loo_no_rdt_v2": {"adadae": {"use_rdt_v2": False}},
    "loo_no_aether": {"adadae": {"use_aether": False}},
    "loo_no_chronos": {"adadae": {"use_chronos": False}},
}

LEAVE_ONE_OUT_A4 = {
    "loo_no_omni": {"adadae": {"use_omni": False, "auto_regime_gates": False}},
    "loo_no_nano": {"adadae": {"use_nano": False}},
    "loo_no_torrent": {"adadae": {"use_torrent": False}},
    "loo_no_prism": {"adadae": {"use_prism": False}},
    "loo_no_polis": {"adadae": {"use_polis": False}},
    "loo_no_sieve": {"adadae": {"use_sieve": False}},
    "loo_no_needle": {"adadae": {"use_needle": False}},
    "loo_no_sparse": {"adadae": {"use_sparse_view": False}},
    "loo_no_robust": {"adadae": {"use_robust": False}},
    "loo_no_quell": {"adadae": {"use_quell": False, "fusion_mode": "calix"}},
    "loo_no_geode": {"adadae": {"use_geode": False}},
    "loo_no_helios": {"adadae": {"use_helios": False}},
}

LEAVE_ONE_OUT_A5 = {
    "loo_no_figaro": {"adadae": {"use_figaro": False}},
    "loo_no_dsm_plus": {"adadae": {"use_dsm_plus": False}},
    "loo_no_mahala": {"adadae": {"use_mahala": False}},
    "loo_no_full_dte": {"adadae": {"use_full_dte": False}},
    "loo_no_lexicon": {"adadae": {"use_lexicon": False, "fusion_mode": "calibrated"}},
    "loo_no_pura": {"adadae": {"use_pura": False}},
    "loo_no_evt": {"adadae": {"use_evt_tail": False}},
    "loo_no_confal": {"adadae": {"use_confal": False}},
    "loo_no_spectra": {"adadae": {"use_spectra": False}},
    "loo_no_sinkhorn": {"adadae": {"use_sinkhorn": False}},
    "loo_no_ib": {"adadae": {"use_ib_latent": False}},
    "loo_no_elbo": {"adadae": {"use_elbo_s": False}},
    "loo_no_curriculum": {"adadae": {"use_curriculum_snr": False}},
    "loo_no_vmf": {"adadae": {"use_vmf_z": False}},
}

LEAVE_ONE_OUT_A6 = {
    "loo_no_helix": {"adadae": {"use_helix": False}},
    "loo_no_delta": {"adadae": {"use_delta": False}},
    "loo_no_apex": {"adadae": {"use_apex": False}},
    "loo_no_nautilus": {"adadae": {"use_nautilus": False}},
    "loo_no_torque": {"adadae": {"use_torque": False}},
    "loo_no_orbit": {"adadae": {"use_orbit": False}},
    "loo_no_kale": {"adadae": {"use_kale": False, "use_lexicon": True, "fusion_mode": "lexicon"}},
    "loo_no_ridge": {"adadae": {"use_ridge": False}},
    "loo_no_locus": {"adadae": {"use_locus": False}},
    "loo_no_spiral": {"adadae": {"use_spiral": False}},
}


def deep_update(base: dict, overrides: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ablation_ladder.yaml")
    parser.add_argument(
        "--hardware",
        type=str,
        default=None,
        help="Override hardware profile: 8gb|12gb|16gb|rtx5070ti",
    )
    parser.add_argument("--setting", default="semi-supervised")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--full", action="store_true")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["cardio", "thyroid", "breastw", "pendigits", "vowels"],
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--steps",
        nargs="*",
        default=None,
        help="Subset of ablation steps (default: full ladder)",
    )
    parser.add_argument(
        "--leave-one-out",
        action="store_true",
        help="Run leave-one-component-out from full_adadae2",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    args = parser.parse_args()

    base = load_config(args.config, hardware=args.hardware)
    base["train"]["epochs"] = args.epochs
    base["train"]["eval_every"] = max(5, args.epochs // 5)
    base["paths"]["run_id"] = "ablations"

    adbench = Path(base["paths"]["adbench_root"])
    registry = build_registry(adbench)
    if not args.full:
        wanted = {d.lower() for d in args.datasets}
        registry = [s for s in registry if s.name.lower() in wanted]

    if args.leave_one_out:
        cfg_l = str(args.config).lower()
        if "adadae6" in cfg_l:
            full_key, loo_map = "full_adadae6", LEAVE_ONE_OUT_A6
        elif "adadae5" in cfg_l:
            full_key, loo_map = "full_adadae5", LEAVE_ONE_OUT_A5
        elif "adadae4" in cfg_l:
            full_key, loo_map = "full_adadae4", LEAVE_ONE_OUT_A4
        elif "adadae3" in cfg_l:
            full_key, loo_map = "full_adadae3", LEAVE_ONE_OUT_A3
        else:
            full_key, loo_map = "full_adadae2", LEAVE_ONE_OUT
        if full_key not in ABLATIONS:
            raise SystemExit(f"missing ablation key {full_key} for leave-one-out")
        full = deep_update(base, ABLATIONS[full_key])
        steps_map = {
            name: deep_update(full, ov) for name, ov in loo_map.items()
        }
        steps = list(steps_map.keys())
    else:
        steps_map = None
        steps = args.steps or LADDER_ORDER

    seeds = args.seeds or [args.seed]
    results_dir = Path(base["paths"]["results_dir"])
    setting_slug = args.setting.replace("-", "_")
    logger = RunLogger(results_dir / "logs" / f"ablations_{setting_slug}.jsonl", run_id="ablations")
    out = {}

    for abl_name in steps:
        if steps_map is not None:
            cfg = steps_map[abl_name]
        else:
            if abl_name not in ABLATIONS:
                raise SystemExit(f"Unknown ablation step: {abl_name}")
            cfg = deep_update(base, ABLATIONS[abl_name])
        logger.info(f"Ablation {abl_name}", n_datasets=len(registry))
        ds_metrics = []
        for spec in registry:
            for seed in seeds:
                split_rows = []
                for rel in spec.relative_paths:
                    if not args.full and len(spec.relative_paths) > 1 and rel != spec.relative_paths[0]:
                        continue
                    row = run_single_file(
                        npz_path=adbench / rel,
                        setting=args.setting,
                        seed=seed,
                        config=cfg,
                        logger=logger,
                        dataset_name=spec.name,
                        split_name=rel,
                        category=spec.category,
                    )
                    split_rows.append(row)
                    cleanup_memory()
                agg = mean_std_metrics([r["metrics"] for r in split_rows])
                ds_metrics.append({k: v["mean"] for k, v in agg.items()})
        overall = mean_std_metrics(ds_metrics)
        out[abl_name] = {
            "setting": args.setting,
            "seeds": seeds,
            "n_datasets": len(registry),
            "n_rows": len(ds_metrics),
            "metrics": {k: v["mean"] for k, v in overall.items()},
        }
        print(abl_name, out[abl_name]["metrics"])

    out_dir = results_dir / "thesis"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "loo" if args.leave_one_out else "ladder"
    out_path = out_dir / f"ablation_{suffix}_{setting_slug}.json"
    csv_path = out_dir / f"ablation_{suffix}_{setting_slug}.csv"
    if out_path.exists():
        try:
            prior = json.loads(out_path.read_text(encoding="utf-8"))
            if isinstance(prior, dict):
                prior.update(out)
                out = prior
        except json.JSONDecodeError:
            pass
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    rows = []
    baseline_pr = out.get(
        "ddae_repro",
        out.get(
            "full_adadae5",
            out.get("full_adadae4", out.get("full_adadae3", out.get("full_adadae2", {}))),
        ),
    ).get("metrics", {}).get("PR-AUC", 0.0)
    order = list(steps) if args.leave_one_out else (args.steps or LADDER_ORDER)
    for step in order:
        if step not in out:
            continue
        pr = out[step]["metrics"].get("PR-AUC", 0.0)
        rows.append({
            "step": step,
            "PR-AUC": pr * 100,
            "ROC-AUC": out[step]["metrics"].get("ROC-AUC", 0.0) * 100,
            "delta_PR_vs_ref": (pr - baseline_pr) * 100,
        })
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    logger.close()
    print(f"\nWrote {out_path}")
    if not df.empty:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
