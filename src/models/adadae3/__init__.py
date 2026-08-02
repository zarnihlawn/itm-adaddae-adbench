"""AdaDDAE-3 modules: HELIOS…EPOCHÉ."""
from .aegis import AegisState, apply_aegis, fit_aegis
from .argos import fuse_argos_views
from .atlas import FilmGenerator, MetaPhi, dataset_stats_vector, film
from .epoche import EpocheState, epoche_step
from .flux import FluxHead, flux_loss, flux_residual_score
from .helios import HeliosTimeMap, helios_select_score_grid, sample_helios_t
from .hydra import HydraHeads
from .kairos import kairos_score_budget, kairos_split, kairos_train_T
from .lynx import fuse_lynx_views
from .mirage import mirage_from_residuals, mirage_score
from .nexus_v2 import nexus_v2_loss
from .orbis import orbis_score
from .phasor import PhasorTimeEmbedding
from .plexus import plexus_score
from .rdt_v2 import SoftRejectionEMA, soft_rejection_weights
from .scribe import scribe_ascent, scribe_restore, scribe_should_run
from .strata import strata_consistency_loss, strata_pool

__all__ = [
    "HeliosTimeMap",
    "sample_helios_t",
    "helios_select_score_grid",
    "kairos_train_T",
    "kairos_score_budget",
    "kairos_split",
    "orbis_score",
    "strata_consistency_loss",
    "strata_pool",
    "plexus_score",
    "PhasorTimeEmbedding",
    "fuse_argos_views",
    "AegisState",
    "fit_aegis",
    "apply_aegis",
    "mirage_score",
    "mirage_from_residuals",
    "nexus_v2_loss",
    "SoftRejectionEMA",
    "soft_rejection_weights",
    "fuse_lynx_views",
    "FilmGenerator",
    "MetaPhi",
    "dataset_stats_vector",
    "film",
    "HydraHeads",
    "FluxHead",
    "flux_loss",
    "flux_residual_score",
    "scribe_should_run",
    "scribe_ascent",
    "scribe_restore",
    "EpocheState",
    "epoche_step",
]
