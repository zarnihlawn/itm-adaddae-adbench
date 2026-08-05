"""AdaDDAE: Adaptive Diffusion-Scheduled Denoising Autoencoder (v3)."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ..memory import (
    GuardType,
    MemoryGuard,
    VRAMGuard,
    choose_score_batch_size,
    guard_memory_mb,
    guard_over_limit,
    resolve_amp_dtype,
    shrink_batch,
)
from ..runlog.logger import RunLogger
from .danc import NoiseConfig
from .network import DiffusionBottleneckAE
from .scheduler import DiffusionScheduler
from .scs import select_timesteps, timestep_weights, vectorized_q_sample
from .dte import (
    build_latent_memory,
    fuse_dte_scores,
    knn_dte_score,
    posterior_mean_from_recon,
)
from .fusion_smc import collect_train_view_samples, estimate_view_reliability, fuse_smc_views
from .adadae2 import (
    aether_path_energy,
    dsm_energy_loss,
    fit_calix_weights,
    fuse_calix_views,
    geode_score,
    nexus_ssl_loss,
)
from .adadae3 import (
    AegisState,
    EpocheState,
    FluxHead,
    HeliosTimeMap,
    HydraHeads,
    MetaPhi,
    SoftRejectionEMA,
    apply_aegis,
    dataset_stats_vector,
    epoche_step,
    fit_aegis,
    flux_loss,
    flux_residual_score,
    fuse_argos_views,
    fuse_lynx_views,
    helios_select_score_grid,
    kairos_score_budget,
    kairos_train_T,
    nexus_v2_loss,
    orbis_score,
    plexus_score,
    sample_helios_t,
    scribe_ascent,
    scribe_restore,
    scribe_should_run,
    soft_rejection_weights,
    strata_consistency_loss,
)
from .adadae4 import (
    enrich_omni_meta,
    fit_polis_prototypes,
    fit_prism,
    fit_sieve_iforest,
    fit_sparse_prototype,
    nano_disable_heavy,
    nano_model_dims,
    nano_neighbors,
    nano_weight_decay,
    needle_aegis_alpha,
    needle_tail_boost,
    omni_to_phi,
    polis_score,
    prism_score,
    prism_should_disable_orbis,
    quell_fuse,
    quell_reliability,
    regime_gate_flags,
    robust_normalize_views,
    sieve_iforest_view,
    sieve_rejection_quantile,
    sparse_score,
    torrent_memory_size,
    torrent_score_batch,
    torrent_ssl_subsample,
)
from .adadae5 import (
    confal_apply,
    curriculum_t_max,
    dsm_plus_residual_term,
    elbo_from_residual,
    elbo_score_view,
    evt_tail_transform,
    fit_confal,
    fit_evt_gpd,
    fit_lexicon_weights,
    fit_mahala,
    full_dte_fuse,
    ib_latent_loss,
    lexicon_fuse,
    mahala_score,
    pura_sample_weights,
    sinkhorn_ot_score,
    vmf_concentration_loss,
    vmf_normalize,
)
from .adadae6 import (
    apex_transform,
    fit_apex,
    fit_kale_weights,
    fit_orbit,
    kale_fuse,
    locus_lof_score,
    nautilus_memory_size,
    nautilus_model_dims,
    nautilus_neighbors,
    orbit_score,
    ridge_huber_loss,
    spiral_consistency_score,
    torque_memory_size,
    torque_score_batch,
)


class AdaDDAE:
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        latent_dim: int = 32,
        activation: str = "lrelu",
        noise_config: Optional[NoiseConfig] = None,
        time_emb_type: str = "sinusoidal",
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        device: Optional[torch.device] = None,
        eval_every: int = 10,
        contrastive: bool = True,
        contrastive_alpha: float = 0.2,
        contrastive_gamma: float = 1.0,
        contrastive_adaptive_alpha: bool = True,
        contrastive_pairing: str = "taps",
        hard_negative_mining: bool = True,
        use_scs: bool = True,
        scs_max_timesteps: int = 32,
        scs_mode: str = "snr_weighted",
        scs_selection: str = "snr_stratified",
        use_multiview: bool = True,
        use_uncertainty_view: bool = False,
        uncertainty_draws: int = 3,
        score_noise_draws: int = 1,
        use_dte_view: bool = True,
        dte_knn: int = 5,
        dte_memory_size: int = 4096,
        dte_knn_weight: float = 0.5,
        use_rejection_training: bool = True,
        rejection_quantile: float = 0.95,
        rejection_min_weight: float = 0.1,
        rejection_warmup_epochs: int = 1,
        fusion_mode: str = "fixed",
        fusion_weights: Optional[Dict[str, float]] = None,
        setting: str = "semi-supervised",
        score_batch_size: int = 1024,
        memory_guard: Optional[GuardType] = None,
        early_stop_patience: int = 20,
        min_epochs: int = 0,
        use_amp: bool = False,
        amp_dtype: str = "bfloat16",
        pin_memory: bool = False,
        num_workers: int = 0,
        vectorized_scoring: bool = False,
        preupload_test_threshold: int = 50000,
        y_train: Optional[torch.Tensor] = None,
        # AdaDDAE-2
        use_geode: bool = False,
        geode_neighbors: int = 32,
        geode_components: int = 4,
        use_aether: bool = False,
        aether_loss_weight: float = 0.1,
        use_nexus: bool = False,
        nexus_loss_weight: float = 0.05,
        nexus_noise_std: float = 0.05,
        # AdaDDAE-3
        use_helios: bool = False,
        use_kairos: bool = False,
        use_orbis: bool = False,
        orbis_top_k: int = 8,
        use_strata: bool = False,
        strata_scales: int = 2,
        strata_loss_weight: float = 0.05,
        use_plexus: bool = False,
        plexus_neighbors: int = 16,
        use_phasor: bool = False,
        use_argos: bool = False,
        use_aegis: bool = False,
        aegis_alpha: float = 0.1,
        aegis_temperature: float = 1.0,
        use_mirage: bool = False,
        mirage_draws: int = 5,
        use_nexus_v2: bool = False,
        use_rdt_v2: bool = False,
        rdt_v2_steepness: float = 4.0,
        use_lynx: bool = False,
        use_atlas: bool = False,
        atlas_film_rank: int = 32,
        use_hydra: bool = False,
        use_flux: bool = False,
        flux_loss_weight: float = 0.05,
        use_scribe: bool = False,
        scribe_every_k: int = 5,
        scribe_rho: float = 0.05,
        use_epoche: bool = False,
        epoche_patience: int = 5,
        use_compile: bool = False,
        profile_breakdown: bool = False,
        # AdaDDAE-4
        use_omni: bool = False,
        use_nano: bool = False,
        use_torrent: bool = False,
        use_prism: bool = False,
        use_polis: bool = False,
        use_sieve: bool = False,
        use_needle: bool = False,
        use_sparse_view: bool = False,
        use_robust: bool = False,
        use_quell: bool = False,
        meta_features: Optional[Dict[str, Any]] = None,
        category: str = "classical",
        auto_regime_gates: bool = False,
        n_train: Optional[int] = None,
        # AdaDDAE-5
        use_figaro: bool = False,
        use_dsm_plus: bool = False,
        dsm_plus_lambda: float = 0.5,
        use_mahala: bool = False,
        use_full_dte: bool = False,
        use_lexicon: bool = False,
        use_pura: bool = False,
        use_evt_tail: bool = False,
        use_confal: bool = False,
        use_spectra: bool = False,
        use_sinkhorn: bool = False,
        use_ib_latent: bool = False,
        ib_latent_beta: float = 0.01,
        use_elbo_s: bool = False,
        use_curriculum_snr: bool = False,
        use_vmf_z: bool = False,
        vmf_kappa: float = 1.0,
        # AdaDDAE-6
        use_helix: bool = False,
        use_delta: bool = False,
        use_apex: bool = False,
        use_nautilus: bool = False,
        use_torque: bool = False,
        use_orbit: bool = False,
        use_kale: bool = False,
        use_ridge: bool = False,
        ridge_delta: float = 1.0,
        use_locus: bool = False,
        use_spiral: bool = False,
    ):
        self.input_dim = input_dim
        self.setting = setting
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.eval_every = eval_every
        self.contrastive = contrastive
        self.contrastive_alpha = contrastive_alpha
        self.contrastive_gamma = contrastive_gamma
        self.contrastive_adaptive_alpha = contrastive_adaptive_alpha
        self.contrastive_pairing = contrastive_pairing
        self.hard_negative_mining = hard_negative_mining
        self.use_scs = use_scs
        self.scs_max_timesteps = scs_max_timesteps
        self.scs_mode = scs_mode
        self.scs_selection = scs_selection
        self.use_multiview = use_multiview
        # MIRAGE is CPU-safe epistemic variance; VUS stays CUDA-only.
        self.use_uncertainty_view = bool(use_uncertainty_view) and (
            (device or torch.device("cpu")).type == "cuda" or bool(use_mirage)
        )
        self.uncertainty_draws = max(2, int(uncertainty_draws))
        # Phase 2: multi-ε mean when uncertainty view is off (deterministic seeds).
        self.score_noise_draws = max(1, int(score_noise_draws))
        self.use_dte_view = use_dte_view
        self.dte_knn = max(1, int(dte_knn))
        self.dte_memory_size = int(dte_memory_size)
        self.dte_knn_weight = float(dte_knn_weight)
        self.use_rejection_training = use_rejection_training
        self.rejection_quantile = float(rejection_quantile)
        self.rejection_min_weight = float(rejection_min_weight)
        self.rejection_warmup_epochs = max(1, int(rejection_warmup_epochs))
        self.fusion_mode = fusion_mode
        self.fusion_weights = fusion_weights or {
            "reconstruction": 0.6,
            "latent": 0.3,
            "residual": 0.1,
            "uncertainty": 0.1,
            "diffusion_time": 0.1,
            "geode": 0.1,
            "aether": 0.1,
            "orbis": 0.1,
            "plexus": 0.1,
            "flux": 0.05,
        }
        self.device = device or torch.device("cpu")
        self.score_batch_size = score_batch_size
        self.memory_guard = memory_guard or MemoryGuard()
        self.early_stop_patience = early_stop_patience
        self.min_epochs = max(0, int(min_epochs))
        self.use_amp = use_amp and self.device.type == "cuda"
        self.amp_dtype = resolve_amp_dtype(amp_dtype) if self.use_amp else torch.float32
        self.pin_memory = pin_memory and self.device.type == "cuda"
        self.num_workers = num_workers
        self.vectorized_scoring = vectorized_scoring and self.device.type == "cuda"
        self.preupload_test_threshold = preupload_test_threshold
        self.use_geode = bool(use_geode)
        self.geode_neighbors = max(4, int(geode_neighbors))
        self.geode_components = max(1, int(geode_components))
        self.use_aether = bool(use_aether)
        self.aether_loss_weight = float(aether_loss_weight)
        self.use_nexus = bool(use_nexus)
        self.nexus_loss_weight = float(nexus_loss_weight)
        self.nexus_noise_std = float(nexus_noise_std)
        # AdaDDAE-3 flags
        self.use_helios = bool(use_helios)
        self.use_kairos = bool(use_kairos)
        self.use_orbis = bool(use_orbis)
        self.orbis_top_k = max(1, int(orbis_top_k))
        self.use_strata = bool(use_strata)
        self.strata_scales = max(1, int(strata_scales))
        self.strata_loss_weight = float(strata_loss_weight)
        self.use_plexus = bool(use_plexus)
        self.plexus_neighbors = max(2, int(plexus_neighbors))
        self.use_phasor = bool(use_phasor)
        self.use_argos = bool(use_argos) or fusion_mode == "argos"
        self.use_aegis = bool(use_aegis)
        self.aegis_alpha = float(aegis_alpha)
        self.aegis_temperature = float(aegis_temperature)
        self.use_mirage = bool(use_mirage)
        self.mirage_draws = max(2, int(mirage_draws))
        if self.use_mirage:
            self.use_uncertainty_view = True
            self.uncertainty_draws = max(self.uncertainty_draws, self.mirage_draws)
        self.use_nexus_v2 = bool(use_nexus_v2)
        self.use_rdt_v2 = bool(use_rdt_v2)
        self.rdt_v2_steepness = float(rdt_v2_steepness)
        self.use_lynx = bool(use_lynx) or fusion_mode == "lynx"
        self.use_atlas = bool(use_atlas)
        self.atlas_film_rank = max(8, int(atlas_film_rank))
        self.use_hydra = bool(use_hydra)
        self.use_flux = bool(use_flux)
        self.flux_loss_weight = float(flux_loss_weight)
        self.use_scribe = bool(use_scribe)
        self.scribe_every_k = max(0, int(scribe_every_k))
        self.scribe_rho = float(scribe_rho)
        self.use_epoche = bool(use_epoche)
        self.epoche_patience = max(1, int(epoche_patience))
        self.use_compile = bool(use_compile) and self.device.type == "cuda"
        self.profile_breakdown = bool(profile_breakdown)
        # AdaDDAE-4 regime flags
        self.category = str(category or "classical")
        self.meta_features: Dict[str, Any] = dict(meta_features or {})
        self.use_omni = bool(use_omni)
        self.use_nano = bool(use_nano)
        self.use_torrent = bool(use_torrent)
        self.use_prism = bool(use_prism)
        self.use_polis = bool(use_polis)
        self.use_sieve = bool(use_sieve)
        self.use_needle = bool(use_needle)
        self.use_sparse_view = bool(use_sparse_view)
        self.use_robust = bool(use_robust)
        self.use_quell = bool(use_quell) or fusion_mode == "quell"
        self._prism_state = None
        self._polis_prototypes = None
        self._sparse_proto = None
        self._sieve_clf = None
        self._omni_phi_t: Optional[torch.Tensor] = None
        # AdaDDAE-5
        self.use_figaro = bool(use_figaro)
        self.use_dsm_plus = bool(use_dsm_plus)
        self.dsm_plus_lambda = float(dsm_plus_lambda)
        self.use_mahala = bool(use_mahala)
        self.use_full_dte = bool(use_full_dte)
        self.use_lexicon = bool(use_lexicon) or fusion_mode == "lexicon"
        if self.use_lexicon:
            self.fusion_mode = "lexicon"
        self.use_pura = bool(use_pura)
        self.use_evt_tail = bool(use_evt_tail)
        self.use_confal = bool(use_confal)
        self.use_spectra = bool(use_spectra)
        self.use_sinkhorn = bool(use_sinkhorn)
        if self.use_sinkhorn and int(input_dim) > 64:
            self.use_sinkhorn = False
        self.use_ib_latent = bool(use_ib_latent)
        self.ib_latent_beta = float(ib_latent_beta)
        self.use_elbo_s = bool(use_elbo_s)
        self.use_curriculum_snr = bool(use_curriculum_snr)
        self.use_vmf_z = bool(use_vmf_z)
        self.vmf_kappa = float(vmf_kappa)
        # AdaDDAE-6
        self.use_helix = bool(use_helix)
        self.use_delta = bool(use_delta)
        self.use_apex = bool(use_apex)
        self.use_nautilus = bool(use_nautilus)
        self.use_torque = bool(use_torque)
        self.use_orbit = bool(use_orbit)
        self.use_kale = bool(use_kale) or fusion_mode == "kale"
        if self.use_kale:
            self.fusion_mode = "kale"
        self.use_ridge = bool(use_ridge)
        self.ridge_delta = float(ridge_delta)
        self.use_locus = bool(use_locus)
        self.use_spiral = bool(use_spiral)
        self._mahala_state = None
        self._evt_state = None
        self._confal_state = None
        self._apex_state = None
        self._orbit_state = None
        self._pura_weights: Optional[torch.Tensor] = None
        self._lexicon_weights: Optional[Dict[str, float]] = None
        self._kale_weights: Optional[Dict[str, float]] = None
        n_est = int(n_train) if n_train is not None else int(self.meta_features.get("n", 0) or 0)

        if auto_regime_gates and self.meta_features:
            gates = regime_gate_flags(self.meta_features)
            self.use_nano = self.use_nano or gates.get("use_nano", False)
            self.use_torrent = self.use_torrent or gates.get("use_torrent", False)
            self.use_prism = self.use_prism or gates.get("use_prism", False)
            self.use_polis = self.use_polis or gates.get("use_polis", False)
            self.use_sieve = self.use_sieve or gates.get("use_sieve", False)
            self.use_needle = self.use_needle or gates.get("use_needle", False)
            self.use_sparse_view = self.use_sparse_view or gates.get("use_sparse", False)
            self.use_robust = self.use_robust or gates.get("use_robust", False)
            self.use_geode = self.use_geode or gates.get("use_geode", False)
            self.use_orbis = self.use_orbis or gates.get("use_orbis", False)

        if self.use_nano and n_est > 0:
            for k, v in nano_disable_heavy(n_est).items():
                if hasattr(self, k):
                    setattr(self, k, v and getattr(self, k))
                if k == "use_flux":
                    self.use_flux = False
                if k == "use_mirage":
                    self.use_mirage = False
                    self.use_uncertainty_view = False
            self.geode_neighbors, self.plexus_neighbors = nano_neighbors(
                n_est, self.geode_neighbors, self.plexus_neighbors
            )

        if self.use_torrent and n_est > 0:
            self.dte_memory_size = torrent_memory_size(n_est, self.dte_memory_size)
            self.score_batch_size = torrent_score_batch(n_est, self.score_batch_size)

        if self.use_sieve and self.meta_features:
            self.rejection_quantile = sieve_rejection_quantile(
                float(self.meta_features.get("contamination", 0.05)),
                base=self.rejection_quantile,
            )

        if self.use_needle and self.meta_features:
            self.aegis_alpha = needle_aegis_alpha(
                float(self.meta_features.get("contamination", 0.05)),
                base=self.aegis_alpha,
            )

        if self.use_prism and prism_should_disable_orbis(input_dim, self.category):
            self.use_orbis = False

        if self.use_omni and self.meta_features:
            phi = omni_to_phi(self.meta_features, dim=16)
            self._omni_phi_t = torch.tensor(phi, dtype=torch.float32)

        self._calibrated_fusion: Optional[Dict[str, float]] = None
        self._smc_reliability: Optional[Dict[str, float]] = None
        self._normal_pool: Optional[torch.Tensor] = None
        self._train_z_cache: Optional[torch.Tensor] = None
        self._cached_t_grid: Optional[torch.Tensor] = None
        self._cached_score_weights: Optional[torch.Tensor] = None
        self._cached_alpha_t: Optional[torch.Tensor] = None
        self._dte_memory: Optional[torch.Tensor] = None
        self._sample_weights: Optional[torch.Tensor] = None
        self._calix_train_scores: Optional[torch.Tensor] = None
        self._aegis_state: Optional[AegisState] = None
        self._rdt_v2: Optional[SoftRejectionEMA] = SoftRejectionEMA() if self.use_rdt_v2 else None
        self._epoche: EpocheState = EpocheState()
        self._score_boost: int = 0
        self._kairos_T: Optional[int] = None
        self.timing: Dict[str, float] = {}
        self.profile: Dict[str, float] = {
            "train": 0.0,
            "score": 0.0,
            "geode": 0.0,
            "fusion": 0.0,
            "orbis": 0.0,
            "plexus": 0.0,
        }

        if y_train is not None and setting == "semi-supervised":
            normal_mask = (y_train == 0)
            if normal_mask.any():
                self._normal_pool = torch.where(normal_mask)[0]

        nc = noise_config or NoiseConfig(50, "linear", 1e-4, 0.02, 8)
        self.noise_config = nc
        self.T = nc.num_timesteps

        hdims = list(hidden_dims or [512, 512])
        lat = int(latent_dim)
        if self.use_nano and n_est > 0:
            hdims, lat = nano_model_dims(n_est, hdims, lat)
        if self.use_nautilus and n_est > 0:
            hdims, lat = nautilus_model_dims(n_est, hdims, lat)
            gk, dk = nautilus_neighbors(n_est, self.geode_neighbors, self.dte_knn)
            self.geode_neighbors, self.dte_knn = gk, dk
            self.dte_memory_size = nautilus_memory_size(n_est, self.dte_memory_size)
        if self.use_torque and n_est > 0:
            self.dte_memory_size = torque_memory_size(n_est, self.dte_memory_size)
            self.score_batch_size = torque_score_batch(n_est, self.score_batch_size)

        emb_type = "phasor" if self.use_phasor else time_emb_type
        self.model = DiffusionBottleneckAE(
            input_dim=input_dim,
            hidden_dims=hdims,
            latent_dim=lat,
            time_emb_dim=nc.time_emb_dim,
            activation=activation,
            time_emb_type=emb_type,
            use_atlas=self.use_atlas,
            atlas_cond_dim=24,
            atlas_film_hidden=self.atlas_film_rank,
        ).to(self.device)

        self.helios: Optional[HeliosTimeMap] = None
        if self.use_helios:
            self.helios = HeliosTimeMap(hidden=16).to(self.device)

        self.meta_phi: Optional[MetaPhi] = None
        if self.use_atlas:
            self.meta_phi = MetaPhi(in_dim=16 if self.use_omni else 8, out_dim=16).to(self.device)

        self.hydra: Optional[HydraHeads] = None
        if self.use_hydra:
            self.hydra = HydraHeads(latent_dim=lat, input_dim=input_dim).to(self.device)

        self.flux_head: Optional[FluxHead] = None
        if self.use_flux:
            self.flux_head = FluxHead(input_dim=input_dim, hidden=128, time_dim=8).to(self.device)

        self.scheduler = DiffusionScheduler(
            num_timesteps=nc.num_timesteps,
            device=self.device,
            beta_start=nc.beta_start,
            beta_end=nc.beta_end,
            scheduler=nc.scheduler,
        )

        selection = scs_selection if use_scs and scs_mode != "full_sum" else "linspace"
        if use_scs and scs_mode != "full_sum":
            self.score_timesteps = select_timesteps(
                self.T,
                setting,
                scs_max_timesteps,
                mode=scs_mode,
                selection=selection,
                alpha_bar=self.scheduler.alpha_bar,
            )
        elif scs_mode == "full_sum":
            self.score_timesteps = list(range(1, self.T))
        else:
            self.score_timesteps = list(range(1, self.T))

        if self.use_helios:
            self.score_timesteps = helios_select_score_grid(
                self.T,
                kairos_score_budget(self.T, scs_max_timesteps) if self.use_kairos else min(scs_max_timesteps, self.T - 1),
                alpha_bar=self.scheduler.alpha_bar,
            )

        self.score_weights = timestep_weights(
            self.score_timesteps,
            setting,
            self.T,
            mode=scs_mode,
            alpha_bar=self.scheduler.alpha_bar,
        )
        self._cache_score_tensors()

        self._scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp and self.amp_dtype == torch.float16,
        )

        if self.use_compile:
            try:
                self.model = torch.compile(self.model)  # type: ignore[assignment]
            except Exception:
                self.use_compile = False

    def _cache_score_tensors(self) -> None:
        self._cached_t_grid = torch.tensor(
            self.score_timesteps, device=self.device, dtype=torch.long
        )
        self._cached_score_weights = torch.tensor(
            self.score_weights, device=self.device, dtype=torch.float32
        )
        # fp32 alpha_bar slice for scoring / GEODE-adjacent math
        ab = self.scheduler.alpha_bar.float()
        idx = (self._cached_t_grid - 1).clamp(0, ab.numel() - 1)
        self._cached_alpha_t = ab[idx].detach()

    def _extra_train_params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        if self.helios is not None:
            params.extend(list(self.helios.parameters()))
        if self.meta_phi is not None:
            params.extend(list(self.meta_phi.parameters()))
        if self.hydra is not None:
            params.extend(list(self.hydra.parameters()))
        if self.flux_head is not None:
            params.extend(list(self.flux_head.parameters()))
        return params

    def _init_atlas_cond(self, x_train: torch.Tensor) -> None:
        if not self.use_atlas or self.meta_phi is None:
            return
        with torch.no_grad():
            if self._omni_phi_t is not None:
                stats = self._omni_phi_t.to(self.device)
            else:
                stats = dataset_stats_vector(x_train[: min(2048, x_train.size(0))].to(self.device))
            phi = self.meta_phi(stats)
            pad = torch.zeros(max(0, 24 - phi.numel()), device=self.device)
            cond = torch.cat([phi.reshape(-1), pad], dim=0)[:24]
            self.model.set_atlas_cond(cond)

    def _fit_adadae4_aux(self, x_train: torch.Tensor) -> None:
        """Fit PRISM / POLIS / SPARSE / SIEVE auxiliaries on train (normals preferred)."""
        if self._normal_pool is not None and self._normal_pool.numel() > 0:
            src = x_train[self._normal_pool]
        else:
            src = x_train
        xb = src[: min(8000, src.size(0))].detach().cpu().numpy()
        if self.use_prism:
            self._prism_state = fit_prism(xb)
        if self.use_sparse_view:
            self._sparse_proto = fit_sparse_prototype(xb)
        if self.use_sieve:
            try:
                self._sieve_clf = fit_sieve_iforest(xb)
            except Exception:
                self._sieve_clf = None
        if self.use_polis and self._dte_memory is not None:
            z = self._dte_memory.detach().cpu().numpy()
            sep = float(self.meta_features.get("cluster_sep", 0.0))
            self._polis_prototypes = fit_polis_prototypes(z, cluster_sep=sep)

    @torch.inference_mode()
    def _compute_pura_weights(self, x_train: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        n = x_train.size(0)
        losses = []
        bs = min(self.batch_size, 512)
        t1 = torch.ones(1, device=self.device, dtype=torch.long)
        for i in range(0, n, bs):
            xb = x_train[i : i + bs].to(self.device)
            x0_hat, _ = self.model(xb, t1.expand(xb.size(0)))
            rec = nn.functional.mse_loss(x0_hat, xb, reduction="none").mean(dim=1)
            losses.append(rec.cpu())
        all_rec = torch.cat(losses, dim=0)
        pi = float(self.meta_features.get("contamination", 0.05) or 0.05)
        w = pura_sample_weights(all_rec, prior_pi=pi)
        self.model.train()
        return w

    @torch.inference_mode()
    def _fit_adadae5_aux(self, x_train: torch.Tensor) -> None:
        """Fit MAHALA on train residuals/latents (train-only)."""
        if not self.use_mahala and not self.use_orbit:
            return
        self.model.eval()
        if self._normal_pool is not None and self._normal_pool.numel() > 0:
            src = x_train[self._normal_pool]
        else:
            src = x_train
        n = min(4096, src.size(0))
        xb = src[:n].to(self.device)
        t0 = torch.ones(n, device=self.device, dtype=torch.long)
        x0_hat, z0 = self.model(xb, t0)
        if self.use_vmf_z:
            z0 = vmf_normalize(z0)
        if self.use_mahala:
            residual = (xb - x0_hat).float()
            self._mahala_state = fit_mahala(
                residual.cpu().numpy(),
                z0.float().cpu().numpy(),
            )
        if self.use_orbit:
            self._orbit_state = fit_orbit(xb.float().cpu().numpy())
        self.model.train()

    @torch.inference_mode()
    def _fit_adadae5_score_calib(self, x_train: torch.Tensor, n_cal: int = 512) -> None:
        """Fit EVT/CONFAL/APEX on fused train scores (same units as predict)."""
        if not (self.use_evt_tail or self.use_confal or self.use_apex):
            return
        if self._normal_pool is not None and self._normal_pool.numel() > 0:
            src = x_train[self._normal_pool]
        else:
            src = x_train
        n = min(n_cal, src.size(0))
        xb = src[:n]
        was_evt, was_conf, was_apex = self.use_evt_tail, self.use_confal, self.use_apex
        self.use_evt_tail = False
        self.use_confal = False
        self.use_apex = False
        try:
            fused = self.predict(xb).detach().cpu().numpy()
        finally:
            self.use_evt_tail = was_evt
            self.use_confal = was_conf
            self.use_apex = was_apex
        # Fit order matches predict: EVT → CONFAL → APEX
        cur = fused
        if was_evt:
            self._evt_state = fit_evt_gpd(cur)
            if self._evt_state is not None:
                cur = (
                    evt_tail_transform(torch.tensor(cur, dtype=torch.float32), self._evt_state)
                    .detach()
                    .cpu()
                    .numpy()
                )
        if was_conf:
            self._confal_state = fit_confal(cur)
            if self._confal_state is not None:
                cur = (
                    confal_apply(torch.tensor(cur, dtype=torch.float32), self._confal_state)
                    .detach()
                    .cpu()
                    .numpy()
                )
        if was_apex:
            c_hat = float(self.meta_features.get("contamination", 0.05) if self.meta_features else 0.05)
            if self.noise_config.contamination_est is not None:
                c_hat = float(self.noise_config.contamination_est)
            self._apex_state = fit_apex(cur, contamination=c_hat)

    def _snr_weight_at_t(self, t: torch.Tensor) -> torch.Tensor:
        """Per-sample SNR weight for TAPS contrastive scaling."""
        ab = self.scheduler.alpha_bar
        idx = t.clamp(0, ab.numel() - 1)
        ab_t = ab[idx]
        if self.setting == "unsupervised":
            w = 1.0 - ab_t
        else:
            w = ab_t
        return w.clamp(min=1e-6)

    def _sample_positive_z(self, z0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """TAPS: semi normal-only positives from cache; unsupervised same-batch pairing."""
        bsz = z0.size(0)
        if self.contrastive_pairing == "random":
            return z0[torch.randperm(bsz, device=self.device)]

        if (
            self.setting == "semi-supervised"
            and self._train_z_cache is not None
            and self._train_z_cache.numel() > 0
        ):
            pool_pos = torch.randint(0, self._train_z_cache.size(0), (bsz,), device=self.device)
            return self._train_z_cache[pool_pos]

        return z0

    def _contrastive_loss_ddae_c(
        self,
        z0: torch.Tensor,
        zt: torch.Tensor,
        t: torch.Tensor,
        z_pos: torch.Tensor,
    ) -> torch.Tensor:
        """DDAE-C Eq. 7: margin m = 1 + gamma * t/T."""
        delta_neg = torch.norm(z0 - zt, dim=1)
        m = 1.0 + self.contrastive_gamma * (t.float() / max(self.T, 1))
        loss_neg = torch.clamp(m - delta_neg, min=0.0) ** 2

        delta_pos = torch.norm(z0 - z_pos, dim=1)
        loss_pos = delta_pos**2

        if self.hard_negative_mining:
            hard_w = torch.softmax(-delta_neg.detach(), dim=0) * z0.size(0)
            loss_neg = loss_neg * hard_w

        return (loss_pos + loss_neg).mean()

    def _effective_contrastive_alpha(self, t: torch.Tensor) -> float:
        if not self.contrastive_adaptive_alpha:
            base = self.contrastive_alpha
        else:
            base = self.contrastive_alpha * (t.float().mean() / max(self.T, 1))
        snr_w = self._snr_weight_at_t(t).mean()
        alpha_t = base * snr_w
        return float(alpha_t.clamp(0.0, 0.5).item())

    def _train_step(
        self,
        x_0: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        sample_idx: Optional[torch.Tensor] = None,
        epoch: int = 0,
    ) -> float:
        bsz = x_0.size(0)
        t_max = self._kairos_T if (self.use_kairos and self._kairos_T is not None) else self.T
        if self.use_curriculum_snr:
            t_max = curriculum_t_max(epoch, self.epochs, int(t_max))
        t_max = max(2, int(t_max))
        if self.use_helios and self.helios is not None:
            t = sample_helios_t(self.helios, bsz, t_max, self.device)
        else:
            t = torch.randint(1, t_max, (bsz,), device=self.device).long()
        t0 = torch.ones(bsz, device=self.device, dtype=torch.long)

        batch_w = None
        if self._sample_weights is not None and sample_idx is not None:
            # _sample_weights stays on CPU; indices must match indexed tensor device.
            batch_w = self._sample_weights[sample_idx.cpu()].to(self.device)
        if self._pura_weights is not None and sample_idx is not None:
            pw = self._pura_weights[sample_idx.cpu()].to(self.device)
            batch_w = pw if batch_w is None else (batch_w * pw)

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        ):
            x_t, noise = self.scheduler.q_sample(x_0, t)
            x0_hat, zt = self.model(x_t, t)
            _, z0 = self.model(x_0, t0)
            if self.use_vmf_z:
                z0 = vmf_normalize(z0)
                zt = vmf_normalize(zt)

            if self.use_dsm_plus:
                rec_ps = dsm_plus_residual_term(
                    x0_hat, x_0, x_t, noise, t, self.scheduler, lambda_dsm=self.dsm_plus_lambda
                )
            elif self.use_ridge:
                rec_ps = ridge_huber_loss(x0_hat, x_0, delta=self.ridge_delta, reduction="none")
            else:
                rec_ps = nn.functional.mse_loss(x0_hat, x_0, reduction="none").mean(dim=1)
            if self.use_ridge and self.use_dsm_plus:
                # Blend robust recon into DSM+ path
                rec_ps = 0.7 * rec_ps + 0.3 * ridge_huber_loss(
                    x0_hat, x_0, delta=self.ridge_delta, reduction="none"
                )

            if self.contrastive and self.contrastive_alpha > 0:
                z_pos = self._sample_positive_z(z0, t)
                if self.use_vmf_z:
                    z_pos = vmf_normalize(z_pos)
                cont_ps = self._contrastive_loss_per_sample(z0, zt, t, z_pos)
                alpha_eff = self._effective_contrastive_alpha(t)
                loss_ps = (1 - alpha_eff) * rec_ps + alpha_eff * cont_ps
            else:
                loss_ps = rec_ps

            if batch_w is not None:
                loss = (loss_ps * batch_w).sum() / batch_w.sum().clamp(min=1e-8)
            else:
                loss = loss_ps.mean()

            if self.use_aether and self.aether_loss_weight > 0:
                loss = loss + self.aether_loss_weight * dsm_energy_loss(
                    self.model, self.scheduler, x_0, t
                )
            if self.use_nexus and self.nexus_loss_weight > 0 and not self.use_nexus_v2:
                loss = loss + self.nexus_loss_weight * nexus_ssl_loss(
                    self.model,
                    x_0,
                    t0,
                    noise_std=self.nexus_noise_std,
                )
            if self.use_nexus_v2 and self.nexus_loss_weight > 0:
                noise_x = x_0 + self.nexus_noise_std * torch.randn_like(x_0)
                _, z1 = self.model(x_0, t0)
                _, z2 = self.model(noise_x, t0)
                loss = loss + self.nexus_loss_weight * nexus_v2_loss(z1, z2)
            if self.use_strata and self.strata_loss_weight > 0:
                loss = loss + self.strata_loss_weight * strata_consistency_loss(
                    z0, zt, scales=self.strata_scales
                )
            if self.use_flux and self.flux_head is not None and self.flux_loss_weight > 0:
                t_norm = (t.float() / max(self.T, 1)).clamp(0.0, 1.0)
                loss = loss + self.flux_loss_weight * flux_loss(
                    self.flux_head, x_0, noise, t_norm
                )
            if self.use_hydra and self.hydra is not None:
                heads = self.hydra(z0)
                loss = loss + 0.05 * nn.functional.mse_loss(heads["recon"], x_0)
            if self.use_ib_latent:
                loss = loss + 0.1 * ib_latent_loss(z0, beta=self.ib_latent_beta)
            if self.use_vmf_z:
                loss = loss + 0.05 * vmf_concentration_loss(z0, kappa=self.vmf_kappa)

        do_scribe = self.use_scribe and scribe_should_run(epoch, self.scribe_every_k)
        optimizer.zero_grad(set_to_none=True)
        if self._scaler.is_enabled():
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            if do_scribe:
                backups = scribe_ascent(self.model.parameters(), rho=self.scribe_rho)
                # second forward for SAM-lite
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    x_t2, _ = self.scheduler.q_sample(x_0, t)
                    x0_hat2, _ = self.model(x_t2, t)
                    loss2 = nn.functional.mse_loss(x0_hat2, x_0)
                optimizer.zero_grad(set_to_none=True)
                self._scaler.scale(loss2).backward()
                self._scaler.unscale_(optimizer)
                scribe_restore(backups)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            if do_scribe:
                backups = scribe_ascent(self.model.parameters(), rho=self.scribe_rho)
                x_t2, _ = self.scheduler.q_sample(x_0, t)
                x0_hat2, _ = self.model(x_t2, t)
                loss2 = nn.functional.mse_loss(x0_hat2, x_0)
                optimizer.zero_grad(set_to_none=True)
                loss2.backward()
                scribe_restore(backups)
            optimizer.step()

        return float(loss.item())

    def _contrastive_loss_per_sample(
        self,
        z0: torch.Tensor,
        zt: torch.Tensor,
        t: torch.Tensor,
        z_pos: torch.Tensor,
    ) -> torch.Tensor:
        delta_neg = torch.norm(z0 - zt, dim=1)
        m = 1.0 + self.contrastive_gamma * (t.float() / max(self.T, 1))
        loss_neg = torch.clamp(m - delta_neg, min=0.0) ** 2
        delta_pos = torch.norm(z0 - z_pos, dim=1)
        loss_pos = delta_pos**2
        if self.hard_negative_mining:
            hard_w = torch.softmax(-delta_neg.detach(), dim=0) * z0.size(0)
            loss_neg = loss_neg * hard_w
        return loss_pos + loss_neg

    def _build_normal_z_cache(self, x_train: torch.Tensor) -> None:
        if self._normal_pool is None or self._normal_pool.numel() == 0:
            return
        self.model.eval()
        with torch.no_grad():
            x_norm = x_train[self._normal_pool].to(self.device)
            t0 = torch.ones(x_norm.size(0), device=self.device, dtype=torch.long)
            _, z = self.model(x_norm, t0)
            self._train_z_cache = z.detach()
        self.model.train()

    @torch.inference_mode()
    def _compute_rdt_weights(self, x_train: torch.Tensor) -> torch.Tensor:
        """RDT / RDT-v2: down-weight train points with high clean reconstruction loss."""
        self.model.eval()
        n = x_train.size(0)
        losses = []
        bs = min(self.batch_size, 512)
        t1 = torch.ones(1, device=self.device, dtype=torch.long)
        for i in range(0, n, bs):
            xb = x_train[i : i + bs].to(self.device)
            x0_hat, _ = self.model(xb, t1.expand(xb.size(0)))
            rec = nn.functional.mse_loss(x0_hat, xb, reduction="none").mean(dim=1)
            losses.append(rec.cpu())
        all_rec = torch.cat(losses, dim=0)
        if self.use_rdt_v2 and self._rdt_v2 is not None:
            w, self._rdt_v2 = soft_rejection_weights(
                all_rec,
                self._rdt_v2,
                min_weight=self.rejection_min_weight,
                steepness=self.rdt_v2_steepness,
                update=True,
            )
            self.model.train()
            return w
        tau = torch.quantile(all_rec, self.rejection_quantile)
        weights = torch.where(
            all_rec <= tau,
            torch.ones_like(all_rec),
            torch.full_like(all_rec, self.rejection_min_weight),
        )
        self.model.train()
        return weights

    @torch.inference_mode()
    def _build_dte_memory(self, x_train: torch.Tensor) -> None:
        """Build latent memory bank for kNN DTE scoring."""
        self.model.eval()
        zs = []
        bs = min(self.batch_size, 512)
        t0 = torch.ones(1, device=self.device, dtype=torch.long)
        if self._normal_pool is not None and self._normal_pool.numel() > 0:
            src = x_train[self._normal_pool]
        else:
            src = x_train
        for i in range(0, src.size(0), bs):
            xb = src[i : i + bs].to(self.device)
            _, z = self.model(xb, t0.expand(xb.size(0)))
            if self.use_vmf_z:
                z = vmf_normalize(z)
            zs.append(z)
        z_all = torch.cat(zs, dim=0)
        # Clone out of inference tensors so later no_grad / train paths can use the bank
        mem = build_latent_memory(z_all, max_samples=self.dte_memory_size)
        self._dte_memory = mem.detach().clone()
        self.model.train()

    @torch.inference_mode()
    def _val_recon_loss(self, x_val: torch.Tensor) -> float:
        """Label-free validation loss: mid-t reconstruction MSE (mean over val)."""
        self.model.eval()
        if x_val is None or x_val.numel() == 0:
            return float("nan")
        total = 0.0
        n = 0
        bs = max(1, min(self.batch_size, 256))
        t_fixed = max(1, int(self.T) // 2)
        for i in range(0, x_val.size(0), bs):
            xb = x_val[i : i + bs].to(self.device)
            t = torch.full((xb.size(0),), t_fixed, device=self.device, dtype=torch.long)
            x_t, _ = self.scheduler.q_sample(xb, t)
            x0_hat, _ = self.model(x_t, t)
            loss = nn.functional.mse_loss(x0_hat, xb, reduction="mean")
            total += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        self.model.train()
        return total / max(n, 1)

    def fit(
        self,
        x_train: torch.Tensor,
        x_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        logger: Optional[RunLogger] = None,
        eval_fn=None,
        early_stop_metric: str = "val_loss",
        *,
        x_test: Optional[torch.Tensor] = None,
        y_test: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Train with early stopping on validation only (never on test labels).

        ``early_stop_metric``:
          - ``val_loss`` (default): minimize reconstruction MSE on ``x_val``
          - ``val_pr``: maximize PR-AUC on ``(x_val, y_val)`` when both classes present
          - ``train_loss``: minimize epoch train loss (fallback if no val)

        Deprecated kwargs ``x_test`` / ``y_test`` are rejected if passed for selection —
        pass them only if empty/None. Final test metrics belong outside ``fit``.
        """
        if x_test is not None or y_test is not None:
            raise ValueError(
                "fit() no longer accepts x_test/y_test for checkpoint selection. "
                "Carve a train-only validation set and pass x_val (and y_val only for val_pr)."
            )

        metric_name = str(early_stop_metric or "val_loss").lower()
        if metric_name not in ("val_loss", "val_pr", "train_loss"):
            raise ValueError(f"Unknown early_stop_metric: {early_stop_metric!r}")

        has_val = x_val is not None and x_val.numel() > 0
        if metric_name in ("val_loss", "val_pr") and not has_val:
            metric_name = "train_loss"

        if metric_name == "val_pr":
            if y_val is None or eval_fn is None:
                metric_name = "val_loss" if has_val else "train_loss"
            else:
                y_np = y_val.detach().cpu().numpy()
                if len(np.unique(y_np)) < 2:
                    # Semi train is normals-only — PR-AUC undefined on val slice.
                    metric_name = "val_loss" if has_val else "train_loss"

        minimize = metric_name in ("val_loss", "train_loss")
        best_metric = float("inf") if minimize else -1.0

        t_train_start = time.perf_counter()
        if isinstance(self.memory_guard, VRAMGuard):
            self.memory_guard.reset_peak()

        if self.setting == "semi-supervised" and self.contrastive_pairing == "taps":
            self._build_normal_z_cache(x_train)

        self._init_atlas_cond(x_train)

        use_rdt = self.use_rejection_training or self.use_rdt_v2
        use_idx = use_rdt or self.use_pura
        if use_idx:
            dataset = TensorDataset(x_train, torch.arange(x_train.size(0)))
        else:
            dataset = TensorDataset(x_train)

        opt_params = list(self.model.parameters()) + self._extra_train_params()
        wd = nano_weight_decay(int(self.meta_features.get("n", 0) or 0)) if self.use_nano else 0.0
        optimizer = Adam(opt_params, lr=self.learning_rate, betas=(0.9, 0.999), weight_decay=wd)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

        best_state = None
        patience = 0
        history = []
        cur_batch = self.batch_size

        epoch_bar = tqdm(range(self.epochs), desc="train", leave=False)
        for epoch in epoch_bar:
            if self.use_kairos:
                self._kairos_T = kairos_train_T(epoch, self.epochs, self.T)
            self.model.train()
            total_loss = 0.0
            n_batches = 0

            for batch in loader:
                if use_idx:
                    x_0, batch_idx = batch
                    # Keep indices on CPU for RDT weight lookup.
                else:
                    x_0 = batch[0]
                    batch_idx = None

                if self.pin_memory:
                    x_0 = x_0.to(self.device, non_blocking=True)
                else:
                    x_0 = x_0.to(self.device)

                total_loss += self._train_step(x_0, optimizer, sample_idx=batch_idx, epoch=epoch)
                n_batches += 1

                mem, over = guard_over_limit(self.memory_guard)
                if over:
                    new_bs = shrink_batch(cur_batch)
                    if new_bs < cur_batch and logger:
                        logger.log(
                            "memory_shrink",
                            mem_mb=mem,
                            vram_mb=mem if isinstance(self.memory_guard, VRAMGuard) else None,
                            rss_mb=mem if isinstance(self.memory_guard, MemoryGuard) else None,
                            old_batch=cur_batch,
                            new_batch=new_bs,
                        )
                    cur_batch = new_bs
                    self.batch_size = new_bs

            avg_loss = total_loss / max(n_batches, 1)
            mem = guard_memory_mb(self.memory_guard)
            rss = self.memory_guard.rss_mb() if isinstance(self.memory_guard, MemoryGuard) else None
            vram = self.memory_guard.vram_mb() if isinstance(self.memory_guard, VRAMGuard) else None
            postfix = {"loss": f"{avg_loss:.4f}"}
            if vram is not None:
                postfix["vram_mb"] = f"{vram:.0f}"
            if rss is not None:
                postfix["rss_mb"] = f"{rss:.0f}"
            epoch_bar.set_postfix(**postfix)

            metrics: Dict[str, Any] = {}
            if (epoch + 1) % self.eval_every == 0:
                if metric_name == "train_loss":
                    cur = float(avg_loss)
                    metrics["val_metric"] = cur
                    metrics["early_stop_metric"] = "train_loss"
                elif metric_name == "val_loss":
                    cur = self._val_recon_loss(x_val)  # type: ignore[arg-type]
                    metrics["val_loss"] = cur
                    metrics["val_metric"] = cur
                    metrics["early_stop_metric"] = "val_loss"
                    postfix["val_loss"] = f"{cur:.4f}"
                    epoch_bar.set_postfix(**postfix)
                    if self.use_epoche and np.isfinite(cur):
                        did, lr_fac, boost = epoche_step(
                            self._epoche, float(cur), patience=self.epoche_patience
                        )
                        self._score_boost = boost
                        if did and lr_fac < 1.0:
                            for pg in optimizer.param_groups:
                                pg["lr"] = float(pg["lr"]) * lr_fac
                            if logger:
                                logger.log("epoche_lr_shrink", epoch=epoch + 1, lr=optimizer.param_groups[0]["lr"], score_boost=boost)
                        if self.use_kairos:
                            budget = kairos_score_budget(self.T, self.scs_max_timesteps, boost)
                            if self.use_helios:
                                self.score_timesteps = helios_select_score_grid(
                                    self.T, budget, alpha_bar=self.scheduler.alpha_bar
                                )
                            else:
                                self.score_timesteps = select_timesteps(
                                    self.T,
                                    self.setting,
                                    budget,
                                    mode=self.scs_mode,
                                    selection=self.scs_selection,
                                    alpha_bar=self.scheduler.alpha_bar,
                                )
                            self.score_weights = timestep_weights(
                                self.score_timesteps,
                                self.setting,
                                self.T,
                                mode=self.scs_mode,
                                alpha_bar=self.scheduler.alpha_bar,
                            )
                            self._cache_score_tensors()
                else:
                    if self.fusion_mode in ("calibrated", "smc", "argos", "lynx"):
                        self._calibrate_fusion(x_train)
                    scores = self.predict(x_val)  # type: ignore[arg-type]
                    y_np = y_val.detach().cpu().numpy()  # type: ignore[union-attr]
                    metrics = eval_fn(scores.detach().cpu().numpy(), y_np)
                    cur = float(metrics.get("PR-AUC", float("nan")))
                    metrics["val_metric"] = cur
                    metrics["early_stop_metric"] = "val_pr"
                    postfix["val_pr"] = f"{cur:.4f}"
                    epoch_bar.set_postfix(**postfix)

                improved = False
                if np.isfinite(cur):
                    if minimize and cur < best_metric:
                        improved = True
                    elif (not minimize) and cur > best_metric:
                        improved = True
                if improved:
                    best_metric = cur
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1

            row = {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "rss_mb": rss,
                "vram_mb": vram,
                "batch_size": self.batch_size,
                **metrics,
            }
            history.append(row)
            if logger:
                logger.log(
                    "epoch",
                    setting=self.setting,
                    T=self.T,
                    scheduler=self.noise_config.scheduler,
                    **{k: v for k, v in row.items() if v is not None},
                )

            reached_min = (epoch + 1) >= max(0, int(getattr(self, "min_epochs", 0) or 0))
            if (
                reached_min
                and patience >= max(1, self.early_stop_patience // max(1, self.eval_every))
            ):
                if logger:
                    logger.log(
                        "early_stop",
                        epoch=epoch + 1,
                        best_val_metric=best_metric,
                        early_stop_metric=metric_name,
                        min_epochs=int(getattr(self, "min_epochs", 0) or 0),
                    )
                break

            if use_rdt and (epoch + 1) == self.rejection_warmup_epochs:
                self._sample_weights = self._compute_rdt_weights(x_train)
                if logger:
                    n_rej = int((self._sample_weights < 1.0).sum().item())
                    logger.log(
                        "rdt_update",
                        epoch=epoch + 1,
                        n_rejected=n_rej,
                        quantile=self.rejection_quantile,
                        rdt_v2=self.use_rdt_v2,
                    )

            if self.use_pura and (
                (epoch + 1) == self.rejection_warmup_epochs
                or ((epoch + 1) % max(1, self.eval_every) == 0 and (epoch + 1) >= self.rejection_warmup_epochs)
            ):
                self._pura_weights = self._compute_pura_weights(x_train)
                if logger and (epoch + 1) == self.rejection_warmup_epochs:
                    logger.log(
                        "pura_update",
                        epoch=epoch + 1,
                        prior=float(self.meta_features.get("contamination", 0.05) or 0.05),
                    )

            if (self.use_dte_view or self.use_geode or self.use_plexus) and self._dte_memory is None and (epoch + 1) >= self.rejection_warmup_epochs:
                self._build_dte_memory(x_train)

        if best_state is not None:
            self.model.load_state_dict(best_state)

        if self.use_dte_view or self.use_geode or self.use_plexus or self.use_polis or self.use_mahala or self.use_locus:
            self._build_dte_memory(x_train)

        self._fit_adadae4_aux(x_train)
        self._fit_adadae5_aux(x_train)

        if self.fusion_mode in ("calibrated", "smc", "calix", "argos", "lynx", "quell", "lexicon"):
            self._calibrate_fusion(x_train)

        self._fit_adadae5_score_calib(x_train)

        self.timing["train_sec"] = time.perf_counter() - t_train_start
        self.profile["train"] = self.timing["train_sec"]
        return {
            "history": history,
            "best_val_metric": best_metric,
            "best_pr_auc": None if minimize else best_metric,
            "early_stop_metric": metric_name,
            "timing": self.timing,
            "profile": dict(self.profile),
        }

    def _calibrate_fusion(self, x_train: torch.Tensor, n_cal: int = 256) -> None:
        """Calibrate fusion from training normals (calibrated, SMC, CALIX, ARGOS, LYNX, LEXICON)."""
        self.model.eval()
        if self.fusion_mode == "smc":
            # collect_train_view_samples slices x_train in-place; ensure GPU like calibrated path
            x_cal = x_train if x_train.device == self.device else x_train.to(self.device)
            with torch.inference_mode():
                view_samples = collect_train_view_samples(
                    lambda xb, score_seed=0: self._score_views(
                        xb, vectorized=self.vectorized_scoring, score_seed=score_seed
                    ),
                    x_cal,
                    n_cal=n_cal,
                    n_draws=max(2, self.uncertainty_draws),
                    device=self.device,
                )
            active = {k: v for k, v in view_samples.items() if v is not None}
            if not self.use_uncertainty_view:
                active.pop("uncertainty", None)
            if not self.use_dte_view:
                active.pop("diffusion_time", None)
            self._smc_reliability = estimate_view_reliability(active, self.score_weights)
            self._calibrated_fusion = dict(self._smc_reliability)
            return

        n = min(n_cal, x_train.size(0))
        idx = torch.randperm(x_train.size(0), device=x_train.device)[:n]
        xb = x_train[idx].to(self.device)
        with torch.inference_mode():
            rec, lat, res, var, dte = self._score_views(xb, vectorized=self.vectorized_scoring)
            geode = self._geode_view(xb)
            aether = self._aether_view(xb)
            orbis = self._orbis_view(xb)
            plexus = self._plexus_view(xb)
            flux = self._flux_view(xb)
            prism = self._prism_view(xb)
            polis = self._polis_view(xb)
            sparse_v = self._sparse_view(xb)
            sieve_v = self._sieve_view(xb)
            mahala = self._mahala_view(
                xb, rec, lat, x0_hat=getattr(self, "_clean_x0_hat", None), z0=getattr(self, "_clean_z0", None)
            )
            sinkhorn = self._sinkhorn_view(xb, x0_hat=getattr(self, "_clean_x0_hat", None))
            elbo = self._elbo_view(xb, residual=res)
            orbit = self._orbit_view(xb, x0_hat=getattr(self, "_clean_x0_hat", None))
            locus = self._locus_view(xb, z0=getattr(self, "_clean_z0", None))
            spiral = self._spiral_view(xb)

        views = {
            "reconstruction": rec,
            "latent": lat,
            "residual": res,
        }
        if self.use_uncertainty_view:
            views["uncertainty"] = var
        if self.use_dte_view:
            views["diffusion_time"] = dte
        if self.use_geode:
            views["geode"] = geode
        if self.use_aether:
            views["aether"] = aether
        if self.use_orbis:
            views["orbis"] = orbis
        if self.use_plexus:
            views["plexus"] = plexus
        if self.use_flux:
            views["flux"] = flux
        if self.use_prism:
            views["prism"] = prism
        if self.use_polis:
            views["polis"] = polis
        if self.use_sparse_view:
            views["sparse"] = sparse_v
        if self.use_sieve and self._sieve_clf is not None:
            views["sieve"] = sieve_v
        if self.use_mahala:
            views["mahala"] = mahala
        if self.use_sinkhorn:
            views["sinkhorn"] = sinkhorn
        if self.use_elbo_s:
            views["elbo"] = elbo
        if self.use_orbit:
            views["orbit"] = orbit
        if self.use_locus:
            views["locus"] = locus
        if self.use_spiral:
            views["spiral"] = spiral

        if self.fusion_mode == "kale" or self.use_kale:
            self._kale_weights = fit_kale_weights(views)
            self._calibrated_fusion = dict(self._kale_weights)
            return

        if self.fusion_mode == "lexicon" or self.use_lexicon:
            self._lexicon_weights = fit_lexicon_weights(views)
            self._calibrated_fusion = dict(self._lexicon_weights)
            return

        if self.use_robust:
            views = robust_normalize_views(views)

        if self.fusion_mode == "calix":
            self._calibrated_fusion = fit_calix_weights(views, base_weights=self.fusion_weights)
            self._calix_train_scores = fuse_calix_views(views, self._calibrated_fusion).detach().cpu()
        elif self.fusion_mode in ("argos", "lynx", "quell") or self.use_quell:
            eps = 1e-8
            rel = {k: 1.0 / (float(v.mean().item()) + eps) for k, v in views.items()}
            if self.use_quell or self.fusion_mode == "quell":
                rel = quell_reliability(views, base=rel)
            s = sum(rel.values()) or 1.0
            self._calibrated_fusion = {k: v / s for k, v in rel.items()}
            if self.fusion_mode == "argos":
                fused = fuse_argos_views(views, self._calibrated_fusion)
            elif self.fusion_mode == "lynx":
                fused = fuse_lynx_views(views)
            else:
                fused = quell_fuse(views, self._calibrated_fusion)
            self._calix_train_scores = fused.detach().cpu()
        else:
            eps = 1e-8
            parts = {k: 1.0 / (float(v.mean().item()) + eps) for k, v in views.items()}
            s = sum(parts.values())
            self._calibrated_fusion = {k: v / s for k, v in parts.items()}
            fw = self._calibrated_fusion
            fused = sum(fw[k] * views[k] for k in views)
            self._calix_train_scores = fused.detach().cpu()  # type: ignore[union-attr]

        if self.use_aegis and self._calix_train_scores is not None:
            self._aegis_state = fit_aegis(
                self._calix_train_scores,
                alpha=self.aegis_alpha,
                temperature=self.aegis_temperature,
            )

    @torch.inference_mode()
    def _mahala_view(
        self,
        xb: torch.Tensor,
        rec: Optional[torch.Tensor] = None,
        lat: Optional[torch.Tensor] = None,
        x0_hat: Optional[torch.Tensor] = None,
        z0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.use_mahala:
            return torch.zeros(xb.size(0), device=xb.device, dtype=torch.float32)
        if x0_hat is None or z0 is None:
            t0 = torch.ones(xb.size(0), device=self.device, dtype=torch.long)
            x0_hat, z0 = self.model(xb, t0)
            if self.use_vmf_z:
                z0 = vmf_normalize(z0)
        residual = xb - x0_hat
        return mahala_score(residual, z0, self._mahala_state)

    @torch.inference_mode()
    def _sinkhorn_view(
        self,
        xb: torch.Tensor,
        x0_hat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.use_sinkhorn:
            return torch.zeros(xb.size(0), device=xb.device, dtype=torch.float32)
        if x0_hat is None:
            t0 = torch.ones(xb.size(0), device=self.device, dtype=torch.long)
            x0_hat, _ = self.model(xb, t0)
        return sinkhorn_ot_score(xb, x0_hat)

    @torch.inference_mode()
    def _elbo_view(self, xb: torch.Tensor, score_seed: int = 0, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.use_elbo_s:
            return torch.zeros(xb.size(0), device=xb.device, dtype=torch.float32)
        # Prefer residual already computed in _score_views (no second grid)
        if residual is not None:
            return elbo_from_residual(residual)
        return elbo_score_view(
            self.model,
            self.scheduler,
            xb,
            self.score_timesteps,
            weights=self.score_weights,
            score_seed=score_seed,
        )

    @torch.inference_mode()
    def _orbit_view(self, xb: torch.Tensor, x0_hat: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.use_orbit:
            return torch.zeros(xb.size(0), device=xb.device, dtype=torch.float32)
        if x0_hat is None:
            t0 = torch.ones(xb.size(0), device=self.device, dtype=torch.long)
            x0_hat, _ = self.model(xb, t0)
        return orbit_score(xb, x0_hat, self._orbit_state)

    @torch.inference_mode()
    def _locus_view(self, xb: torch.Tensor, z0: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.use_locus:
            return torch.zeros(xb.size(0), device=xb.device, dtype=torch.float32)
        mem = self._dte_memory
        if mem is None or mem.numel() == 0:
            return torch.zeros(xb.size(0), device=xb.device, dtype=torch.float32)
        if z0 is None:
            t0 = torch.ones(xb.size(0), device=self.device, dtype=torch.long)
            _, z0 = self.model(xb, t0)
            if self.use_vmf_z:
                z0 = vmf_normalize(z0)
        return locus_lof_score(z0, mem, k=self.dte_knn)

    @torch.inference_mode()
    def _spiral_view(self, xb: torch.Tensor, score_seed: int = 0) -> torch.Tensor:
        if not self.use_spiral:
            return torch.zeros(xb.size(0), device=xb.device, dtype=torch.float32)
        return spiral_consistency_score(self.model, self.scheduler, xb, t_frac=0.5, score_seed=score_seed)

    @torch.inference_mode()
    def _orbis_view(self, xb: torch.Tensor) -> torch.Tensor:
        if not self.use_orbis:
            return torch.zeros(xb.size(0), device=xb.device, dtype=torch.float32)
        t0 = torch.ones(xb.size(0), device=self.device, dtype=torch.long)
        # mid-t residual for spectral view
        t_mid = torch.full((xb.size(0),), max(1, self.T // 2), device=self.device, dtype=torch.long)
        x_t, _ = self.scheduler.q_sample(xb, t_mid)
        x0_hat, _ = self.model(x_t, t_mid)
        return orbis_score(xb, x0_hat, top_k=self.orbis_top_k)

    @torch.inference_mode()
    def _plexus_view(self, xb: torch.Tensor) -> torch.Tensor:
        if not self.use_plexus:
            return torch.zeros(xb.size(0), device=xb.device, dtype=torch.float32)
        mem = self._dte_memory
        if mem is None or mem.numel() == 0:
            return torch.zeros(xb.size(0), device=xb.device, dtype=torch.float32)
        t0 = torch.ones(xb.size(0), device=self.device, dtype=torch.long)
        _, z0 = self.model(xb, t0)
        # fp32 for neighbor math
        return plexus_score(z0.float(), mem.to(xb.device).float(), n_neighbors=self.plexus_neighbors)

    @torch.inference_mode()
    def _flux_view(self, xb: torch.Tensor) -> torch.Tensor:
        if not self.use_flux or self.flux_head is None:
            return torch.zeros(xb.size(0), device=xb.device, dtype=torch.float32)
        return flux_residual_score(self.flux_head, xb.float(), t_norm=0.5)

    @torch.inference_mode()
    def _prism_view(self, xb: torch.Tensor) -> torch.Tensor:
        if not self.use_prism:
            return torch.zeros(xb.size(0), device=xb.device, dtype=torch.float32)
        return prism_score(xb, self._prism_state)

    @torch.inference_mode()
    def _polis_view(self, xb: torch.Tensor) -> torch.Tensor:
        if not self.use_polis or self._polis_prototypes is None:
            return torch.zeros(xb.size(0), device=xb.device, dtype=torch.float32)
        t0 = torch.ones(xb.size(0), device=self.device, dtype=torch.long)
        _, z0 = self.model(xb, t0)
        return polis_score(z0, self._polis_prototypes)

    @torch.inference_mode()
    def _sparse_view(self, xb: torch.Tensor) -> torch.Tensor:
        if not self.use_sparse_view:
            return torch.zeros(xb.size(0), device=xb.device, dtype=torch.float32)
        return sparse_score(xb, self._sparse_proto)

    @torch.inference_mode()
    def _sieve_view(self, xb: torch.Tensor) -> torch.Tensor:
        if not self.use_sieve or self._sieve_clf is None:
            return torch.zeros(xb.size(0), device=xb.device, dtype=torch.float32)
        return sieve_iforest_view(xb, self._sieve_clf)

    @torch.inference_mode()
    def _geode_view(self, xb: torch.Tensor) -> torch.Tensor:
        if not self.use_geode:
            return torch.zeros(xb.size(0), device=xb.device, dtype=torch.float32)
        mem = self._dte_memory
        if mem is None or mem.numel() == 0:
            return torch.zeros(xb.size(0), device=xb.device, dtype=torch.float32)
        t0 = torch.ones(xb.size(0), device=self.device, dtype=torch.long)
        _, z0 = self.model(xb, t0)
        return geode_score(
            z0,
            mem.to(xb.device),
            n_neighbors=self.geode_neighbors,
            n_components=self.geode_components,
        )

    @torch.inference_mode()
    def _aether_view(self, xb: torch.Tensor, score_seed: int = 0) -> torch.Tensor:
        if not self.use_aether:
            return torch.zeros(xb.size(0), device=xb.device, dtype=torch.float32)
        return aether_path_energy(
            self.model,
            self.scheduler,
            xb,
            self.score_timesteps,
            weights=self.score_weights,
            score_seed=score_seed,
        )

    def _score_views(
        self,
        xb: torch.Tensor,
        vectorized: bool = True,
        score_seed: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return per-sample rec, lat, res, var, dte accumulators."""
        if xb.device != self.device:
            xb = xb.to(self.device, non_blocking=self.device.type == "cuda")
        b = xb.size(0)
        rec = torch.zeros(b, device=self.device, dtype=torch.float32)
        lat = torch.zeros(b, device=self.device, dtype=torch.float32)
        res = torch.zeros(b, device=self.device, dtype=torch.float32)
        var = torch.zeros(b, device=self.device, dtype=torch.float32)
        dte = torch.zeros(b, device=self.device, dtype=torch.float32)

        ts = self.score_timesteps
        ws = self.score_weights
        t0 = torch.ones(b, device=self.device, dtype=torch.long)
        x0_clean, z0 = self.model(xb, t0)
        # Reuse clean forward for MAHALA / SINKHORN (vMF only on stored z for mahala)
        self._clean_x0_hat = x0_clean
        self._clean_z0 = vmf_normalize(z0) if self.use_vmf_z else z0

        M = self.uncertainty_draws if self.use_uncertainty_view else max(1, self.score_noise_draws)
        t_grid = self._cached_t_grid if self._cached_t_grid is not None else torch.tensor(
            ts, device=self.device, dtype=torch.long
        )
        w_tensor = self._cached_score_weights if self._cached_score_weights is not None else torch.tensor(
            ws, device=self.device, dtype=torch.float32
        )

        if vectorized and len(ts) > 1:
            K = len(ts)
            rec_draws = []
            rec_k_last = None
            for m in range(M):
                rng = torch.Generator(device=self.device)
                rng.manual_seed(int(score_seed) + 17 + m * 997)
                noise = torch.randn(b, K, xb.size(1), generator=rng, device=self.device, dtype=xb.dtype)
                x_t = vectorized_q_sample(xb, t_grid, self.scheduler.alpha_bar, noise)

                flat_x = x_t.reshape(b * K, -1)
                flat_t = t_grid.unsqueeze(0).expand(b, K).reshape(-1)
                x0_hat, zt = self.model(flat_x, flat_t)
                x0_hat = x0_hat.view(b, K, -1)
                zt = zt.view(b, K, -1)

                rec_k = torch.norm(xb.unsqueeze(1) - x0_hat, dim=2)
                rec_k_last = rec_k
                lat_k = torch.norm(z0.unsqueeze(1) - zt, dim=2)
                ab_idx = t_grid.clamp(0, self.scheduler.alpha_bar.numel() - 1)
                ab = self.scheduler.alpha_bar[ab_idx].view(1, K, 1)
                implied = (x_t - torch.sqrt(ab) * x0_hat) / torch.sqrt(1.0 - ab + 1e-8)
                res_k = torch.norm(noise - implied, dim=2)

                w = w_tensor.view(1, K)
                rec_draws.append(rec_k)
                rec += (rec_k * w).sum(dim=1) / M
                lat += (lat_k * w).sum(dim=1) / M
                res += (res_k * w).sum(dim=1) / M

            if self.use_uncertainty_view and M > 1:
                stacked = torch.stack(rec_draws, dim=0)
                if self.use_mirage:
                    # MIRAGE: epistemic std across noise seeds (CPU-safe)
                    var_per_t = stacked.std(dim=0)
                else:
                    var_per_t = stacked.var(dim=0, unbiased=False)
                var = (var_per_t * w_tensor.view(1, K)).sum(dim=1)
            if self.use_dte_view and rec_k_last is not None:
                t_grid_dte = t_grid
                if self.use_full_dte:
                    dte = full_dte_fuse(
                        rec_k_last,
                        t_grid_dte,
                        z0,
                        self._dte_memory,
                        knn=self.dte_knn,
                        T=self.T,
                        knn_weight=self.dte_knn_weight,
                    )
                else:
                    post = posterior_mean_from_recon(rec_k_last, t_grid)
                    if self._dte_memory is not None:
                        knn = knn_dte_score(z0, self._dte_memory, k=self.dte_knn, T=self.T)
                        dte = fuse_dte_scores(knn, post, knn_weight=self.dte_knn_weight)
                    else:
                        dte = post
        else:
            rng = torch.Generator(device=self.device)
            rec_draws = []
            rec_k_collect = []
            for m in range(M):
                rng.manual_seed(int(score_seed) + 17 + m * 997)
                rec_m = torch.zeros(b, device=xb.device, dtype=torch.float32)
                rec_t_vec = []
                for ti, t_val in enumerate(ts):
                    w = float(ws[ti])
                    t = torch.full((b,), int(t_val), device=self.device, dtype=torch.long)
                    noise = torch.randn(xb.shape, generator=rng, device=self.device, dtype=xb.dtype)
                    ab = self.scheduler.alpha_bar[t].view(-1, 1)
                    x_t = torch.sqrt(ab) * xb + torch.sqrt(1.0 - ab) * noise
                    x0_hat, zt = self.model(x_t, t)
                    rec_k = torch.norm(xb - x0_hat, dim=1)
                    rec_t_vec.append(rec_k)
                    rec_m += w * rec_k
                    if m == 0:
                        lat += w * torch.norm(z0 - zt, dim=1)
                        implied = (x_t - torch.sqrt(ab) * x0_hat) / torch.sqrt(1.0 - ab + 1e-8)
                        res += w * torch.norm(noise - implied, dim=1)
                rec_draws.append(rec_m)
                rec += rec_m / M
                if m == 0:
                    rec_k_collect = rec_t_vec
            if self.use_uncertainty_view and M > 1:
                stacked = torch.stack(rec_draws, dim=0)
                if self.use_mirage:
                    var = stacked.std(dim=0)
                else:
                    var = stacked.var(dim=0, unbiased=False)
            if self.use_dte_view and rec_k_collect:
                rec_k_stack = torch.stack(rec_k_collect, dim=1)
                t_vec = torch.tensor(ts, device=self.device, dtype=torch.long)
                if self.use_full_dte:
                    dte = full_dte_fuse(
                        rec_k_stack,
                        t_vec,
                        z0,
                        self._dte_memory,
                        knn=self.dte_knn,
                        T=self.T,
                        knn_weight=self.dte_knn_weight,
                    )
                else:
                    post = posterior_mean_from_recon(rec_k_stack, t_vec)
                    if self._dte_memory is not None:
                        knn = knn_dte_score(z0, self._dte_memory, k=self.dte_knn, T=self.T)
                        dte = fuse_dte_scores(knn, post, knn_weight=self.dte_knn_weight)
                    else:
                        dte = post

        return rec, lat, res, var, dte

    def predict(self, x_test: torch.Tensor, score_seed: int = 0) -> torch.Tensor:
        t_score_start = time.perf_counter()
        with torch.inference_mode():
            self.model.eval()
            n = x_test.size(0)
            score_bs = choose_score_batch_size(n, max_batch=self.score_batch_size)
            rec_acc = torch.zeros(n, dtype=torch.float64)
            lat_acc = torch.zeros(n, dtype=torch.float64)
            res_acc = torch.zeros(n, dtype=torch.float64)
            var_acc = torch.zeros(n, dtype=torch.float64)
            dte_acc = torch.zeros(n, dtype=torch.float64)
            geode_acc = torch.zeros(n, dtype=torch.float64)
            aether_acc = torch.zeros(n, dtype=torch.float64)
            orbis_acc = torch.zeros(n, dtype=torch.float64)
            plexus_acc = torch.zeros(n, dtype=torch.float64)
            flux_acc = torch.zeros(n, dtype=torch.float64)
            prism_acc = torch.zeros(n, dtype=torch.float64)
            polis_acc = torch.zeros(n, dtype=torch.float64)
            sparse_acc = torch.zeros(n, dtype=torch.float64)
            sieve_acc = torch.zeros(n, dtype=torch.float64)
            mahala_acc = torch.zeros(n, dtype=torch.float64)
            sinkhorn_acc = torch.zeros(n, dtype=torch.float64)
            elbo_acc = torch.zeros(n, dtype=torch.float64)
            orbit_acc = torch.zeros(n, dtype=torch.float64)
            locus_acc = torch.zeros(n, dtype=torch.float64)
            spiral_acc = torch.zeros(n, dtype=torch.float64)

            preupload = self.device.type == "cuda" and n <= self.preupload_test_threshold
            x_gpu = x_test.to(self.device) if preupload else None

            K = len(self.score_timesteps)
            while K >= 1:
                try:
                    for i in range(0, n, score_bs):
                        xb = x_gpu[i : i + score_bs] if preupload else x_test[i : i + score_bs].to(self.device)
                        rec, lat, res, var, dte = self._score_views(
                            xb,
                            vectorized=self.vectorized_scoring and K == len(self.score_timesteps),
                            score_seed=score_seed,
                        )
                        t_g0 = time.perf_counter()
                        geode = self._geode_view(xb)
                        if self.profile_breakdown:
                            self.profile["geode"] += time.perf_counter() - t_g0
                        aether = self._aether_view(xb, score_seed=score_seed)
                        t_o0 = time.perf_counter()
                        orbis = self._orbis_view(xb)
                        if self.profile_breakdown:
                            self.profile["orbis"] += time.perf_counter() - t_o0
                        t_p0 = time.perf_counter()
                        plexus = self._plexus_view(xb)
                        if self.profile_breakdown:
                            self.profile["plexus"] += time.perf_counter() - t_p0
                        flux = self._flux_view(xb)
                        prism = self._prism_view(xb)
                        polis = self._polis_view(xb)
                        sparse_v = self._sparse_view(xb)
                        sieve_v = self._sieve_view(xb)
                        mahala = self._mahala_view(
                            xb,
                            rec,
                            lat,
                            x0_hat=getattr(self, "_clean_x0_hat", None),
                            z0=getattr(self, "_clean_z0", None),
                        )
                        sinkhorn = self._sinkhorn_view(xb, x0_hat=getattr(self, "_clean_x0_hat", None))
                        elbo = self._elbo_view(xb, score_seed=score_seed, residual=res)
                        orbit = self._orbit_view(xb, x0_hat=getattr(self, "_clean_x0_hat", None))
                        locus = self._locus_view(xb, z0=getattr(self, "_clean_z0", None))
                        spiral = self._spiral_view(xb, score_seed=score_seed)
                        sl = slice(i, i + xb.size(0))
                        rec_acc[sl] += rec.double().cpu()
                        lat_acc[sl] += lat.double().cpu()
                        res_acc[sl] += res.double().cpu()
                        var_acc[sl] += var.double().cpu()
                        dte_acc[sl] += dte.double().cpu()
                        geode_acc[sl] += geode.double().cpu()
                        aether_acc[sl] += aether.double().cpu()
                        orbis_acc[sl] += orbis.double().cpu()
                        plexus_acc[sl] += plexus.double().cpu()
                        flux_acc[sl] += flux.double().cpu()
                        prism_acc[sl] += prism.double().cpu()
                        polis_acc[sl] += polis.double().cpu()
                        sparse_acc[sl] += sparse_v.double().cpu()
                        sieve_acc[sl] += sieve_v.double().cpu()
                        mahala_acc[sl] += mahala.double().cpu()
                        sinkhorn_acc[sl] += sinkhorn.double().cpu()
                        elbo_acc[sl] += elbo.double().cpu()
                        orbit_acc[sl] += orbit.double().cpu()
                        locus_acc[sl] += locus.double().cpu()
                        spiral_acc[sl] += spiral.double().cpu()

                        mem, over = guard_over_limit(self.memory_guard)
                        if over:
                            score_bs = shrink_batch(score_bs, min_batch=32)
                            self.score_batch_size = score_bs
                    break
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and K > 4:
                        K = max(4, K // 2)
                        self.score_timesteps = self.score_timesteps[:K]
                        self.score_weights = self.score_weights[:K]
                        self._cache_score_tensors()
                        if self.device.type == "cuda":
                            torch.cuda.empty_cache()
                    else:
                        raise e

            if not self.use_multiview:
                self.timing["score_sec"] = time.perf_counter() - t_score_start
                self.profile["score"] = self.timing["score_sec"]
                return rec_acc.float()

            def _norm(v: torch.Tensor) -> torch.Tensor:
                m = v.mean().clamp_min(1e-12)
                return v / m

            def _collect_views() -> Dict[str, torch.Tensor]:
                views: Dict[str, torch.Tensor] = {
                    "reconstruction": rec_acc.float(),
                    "latent": lat_acc.float(),
                    "residual": res_acc.float(),
                }
                if self.use_uncertainty_view:
                    views["uncertainty"] = var_acc.float()
                if self.use_dte_view:
                    views["diffusion_time"] = dte_acc.float()
                if self.use_geode:
                    views["geode"] = geode_acc.float()
                if self.use_aether:
                    views["aether"] = aether_acc.float()
                if self.use_orbis:
                    views["orbis"] = orbis_acc.float()
                if self.use_plexus:
                    views["plexus"] = plexus_acc.float()
                if self.use_flux:
                    views["flux"] = flux_acc.float()
                if self.use_prism:
                    views["prism"] = prism_acc.float()
                if self.use_polis:
                    views["polis"] = polis_acc.float()
                if self.use_sparse_view:
                    views["sparse"] = sparse_acc.float()
                if self.use_sieve and self._sieve_clf is not None:
                    views["sieve"] = sieve_acc.float()
                if self.use_mahala:
                    views["mahala"] = mahala_acc.float()
                if self.use_sinkhorn:
                    views["sinkhorn"] = sinkhorn_acc.float()
                if self.use_elbo_s:
                    views["elbo"] = elbo_acc.float()
                if self.use_orbit:
                    views["orbit"] = orbit_acc.float()
                if self.use_locus:
                    views["locus"] = locus_acc.float()
                if self.use_spiral:
                    views["spiral"] = spiral_acc.float()
                if self.use_robust:
                    views = robust_normalize_views(views)
                return views

            t_f0 = time.perf_counter()
            views = _collect_views()
            if self.fusion_mode == "smc" and self._smc_reliability is not None:
                scores = fuse_smc_views(
                    rec_acc.float(),
                    lat_acc.float(),
                    res_acc.float(),
                    var_acc.float() if self.use_uncertainty_view else None,
                    dte_acc.float() if self.use_dte_view else None,
                    self._smc_reliability,
                    use_uncertainty=self.use_uncertainty_view,
                    use_dte=self.use_dte_view,
                ).double()
                if self.use_geode:
                    scores = scores + float(self._calibrated_fusion.get("geode", 0.1) if self._calibrated_fusion else 0.1) * _norm(geode_acc)
                if self.use_aether:
                    scores = scores + float(self._calibrated_fusion.get("aether", 0.1) if self._calibrated_fusion else 0.1) * _norm(aether_acc)
                if self.use_orbis:
                    scores = scores + float(self._calibrated_fusion.get("orbis", 0.1) if self._calibrated_fusion else 0.1) * _norm(orbis_acc)
                if self.use_plexus:
                    scores = scores + float(self._calibrated_fusion.get("plexus", 0.1) if self._calibrated_fusion else 0.1) * _norm(plexus_acc)
            elif self.fusion_mode == "calix" and self._calibrated_fusion is not None:
                scores = fuse_calix_views(views, self._calibrated_fusion).double()
            elif self.fusion_mode == "argos":
                scores = fuse_argos_views(views, self._calibrated_fusion).double()
            elif self.fusion_mode == "lynx":
                scores = fuse_lynx_views(views).double()
            elif self.fusion_mode == "quell" or self.use_quell:
                rel = self._calibrated_fusion or quell_reliability(views)
                scores = quell_fuse(views, rel).double()
            elif self.fusion_mode == "lexicon" or self.use_lexicon:
                scores = lexicon_fuse(views, self._lexicon_weights or self._calibrated_fusion).double()
            elif self.fusion_mode == "kale" or self.use_kale:
                scores = kale_fuse(views, self._kale_weights or self._calibrated_fusion).double()
            else:
                if self._calibrated_fusion is not None:
                    fw = self._calibrated_fusion
                else:
                    fw = self.fusion_weights
                scores = None
                for name, tensor in views.items():
                    w = float(fw.get(name, 0.05 if name not in ("reconstruction", "latent") else fw.get(name, 0.3)))
                    term = w * (_norm(tensor) if not self.use_robust else tensor)
                    scores = term if scores is None else scores + term
                assert scores is not None
                scores = scores.double()

            if self.use_aegis and self._aegis_state is not None:
                scores = apply_aegis(scores.float(), self._aegis_state).double()
            if self.use_needle:
                scores = needle_tail_boost(scores.float()).double()
            if self.use_evt_tail:
                scores = evt_tail_transform(scores.float(), self._evt_state).double()
            if self.use_confal:
                scores = confal_apply(scores.float(), self._confal_state).double()
            if self.use_apex:
                scores = apex_transform(scores.float(), self._apex_state).double()

            scores = torch.nan_to_num(scores.float(), nan=0.0, posinf=0.0, neginf=0.0).double()

            if self.profile_breakdown:
                self.profile["fusion"] += time.perf_counter() - t_f0

        self.timing["score_sec"] = time.perf_counter() - t_score_start
        self.profile["score"] = self.timing["score_sec"]
        return scores.float()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "noise_config": self.noise_config.__dict__,
            "score_timesteps": self.score_timesteps,
            "setting": self.setting,
            "input_dim": self.input_dim,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.model.load_state_dict(state["model"])
