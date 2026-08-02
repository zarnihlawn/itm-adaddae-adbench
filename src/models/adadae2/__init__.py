"""AdaDDAE-2 advanced modules: CHRONOS, GEODE, CALIX, NEXUS, AETHER."""
from .aether import aether_path_energy, dsm_energy_loss
from .calix import calix_conformal_pvalues, fit_calix_weights, fuse_calix_views
from .chronos import ChronosHypernet, chronos_policy, fit_chronos_on_val_proxy, meta_to_phi
from .geode import build_geode_basis, geode_score
from .nexus import nexus_ssl_loss, vicreg_loss

__all__ = [
    "ChronosHypernet",
    "chronos_policy",
    "fit_chronos_on_val_proxy",
    "meta_to_phi",
    "build_geode_basis",
    "geode_score",
    "fit_calix_weights",
    "fuse_calix_views",
    "calix_conformal_pvalues",
    "nexus_ssl_loss",
    "vicreg_loss",
    "dsm_energy_loss",
    "aether_path_energy",
]
