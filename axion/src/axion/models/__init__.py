"""Model registry for harness + AXION."""

from axion.models.axion_model import AxionModel
from axion.models.centroid import CentroidDistanceModel

MODEL_REGISTRY = {
    "centroid_distance": CentroidDistanceModel,
    "axion": AxionModel,
}


_AXION_KEYS = {
    "hidden",
    "latent",
    "depth",
    "mask_rates",
    "score_k",
    "dropout",
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "patience",
    "val_fraction",
    "latch_alpha",
    "latch_alpha_semi",
    "mae_weight",
    "nll_weight",
    "mae_weight_semi",
    "nll_weight_semi",
    "semi_epoch_boost",
    "device",
    "seed",
    "max_train_samples",
}


def build_model(name: str = "axion", **kwargs):
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}; known={list(MODEL_REGISTRY)}")
    cls = MODEL_REGISTRY[name]
    # Drop nulls from YAML
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if name == "centroid_distance":
        allowed = {"whiten", "eps"}
        kwargs = {k: v for k, v in kwargs.items() if k in allowed}
    elif name == "axion":
        kwargs = {k: v for k, v in kwargs.items() if k in _AXION_KEYS}
    return cls(**kwargs)
