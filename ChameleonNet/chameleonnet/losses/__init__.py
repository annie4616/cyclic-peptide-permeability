from .composite import CompositeLoss, composite_loss
from .chameleonic import chameleonic_magnitude_loss
from .triplet import tanimoto_triplet_loss, build_triplets

__all__ = [
    "CompositeLoss",
    "composite_loss",
    "chameleonic_magnitude_loss",
    "tanimoto_triplet_loss",
    "build_triplets",
]
