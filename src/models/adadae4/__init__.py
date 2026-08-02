"""AdaDDAE-4 regime modules: OMNI…QUELL."""
from .nano import nano_disable_heavy, nano_model_dims, nano_neighbors, nano_weight_decay
from .needle import needle_aegis_alpha, needle_tail_boost
from .omni import OMNI_KEYS, enrich_omni_meta, omni_to_phi, regime_gate_flags
from .polis import fit_polis_prototypes, polis_score
from .prism import PrismState, fit_prism, prism_score, prism_should_disable_orbis
from .quell import quell_fuse, quell_reliability
from .robust import mad_normalize, robust_normalize_views
from .sieve import fit_sieve_iforest, sieve_iforest_view, sieve_rejection_quantile
from .sparse import fit_sparse_prototype, sparse_score
from .torrent import (
    torrent_memory_size,
    torrent_reservoir_indices,
    torrent_score_batch,
    torrent_ssl_subsample,
)

__all__ = [
    "OMNI_KEYS",
    "enrich_omni_meta",
    "omni_to_phi",
    "regime_gate_flags",
    "nano_model_dims",
    "nano_neighbors",
    "nano_disable_heavy",
    "nano_weight_decay",
    "torrent_memory_size",
    "torrent_score_batch",
    "torrent_ssl_subsample",
    "torrent_reservoir_indices",
    "PrismState",
    "fit_prism",
    "prism_score",
    "prism_should_disable_orbis",
    "fit_polis_prototypes",
    "polis_score",
    "sieve_rejection_quantile",
    "fit_sieve_iforest",
    "sieve_iforest_view",
    "needle_aegis_alpha",
    "needle_tail_boost",
    "fit_sparse_prototype",
    "sparse_score",
    "mad_normalize",
    "robust_normalize_views",
    "quell_reliability",
    "quell_fuse",
]
