"""AdaDDAE-6 ADBench-informed modules: HELIX…SPIRAL."""
from .apex import ApexState, apex_transform, fit_apex
from .delta import delta_refine_noise, delta_sandwich_contam
from .helix import helix_refine_noise
from .kale import fit_kale_weights, kale_fuse
from .locus import locus_lof_score
from .nautilus import (
    nautilus_memory_size,
    nautilus_model_dims,
    nautilus_neighbors,
    nautilus_should_apply,
)
from .orbit import OrbitState, fit_orbit, orbit_score
from .ridge import ridge_huber_loss
from .spiral import spiral_consistency_score
from .torque import (
    torque_memory_size,
    torque_score_batch,
    torque_should_apply,
    torque_train_subsample,
)

__all__ = [
    "helix_refine_noise",
    "delta_refine_noise",
    "delta_sandwich_contam",
    "ApexState",
    "fit_apex",
    "apex_transform",
    "nautilus_should_apply",
    "nautilus_model_dims",
    "nautilus_neighbors",
    "nautilus_memory_size",
    "torque_should_apply",
    "torque_memory_size",
    "torque_score_batch",
    "torque_train_subsample",
    "OrbitState",
    "fit_orbit",
    "orbit_score",
    "fit_kale_weights",
    "kale_fuse",
    "ridge_huber_loss",
    "locus_lof_score",
    "spiral_consistency_score",
]
