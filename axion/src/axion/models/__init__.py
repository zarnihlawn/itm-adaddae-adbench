"""Model registry for harness + AXION."""

from axion.models.axion_model import AxionModel
from axion.models.centroid import CentroidDistanceModel

MODEL_REGISTRY = {
    "centroid_distance": CentroidDistanceModel,
    "axion": AxionModel,
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
    return cls(**kwargs)
