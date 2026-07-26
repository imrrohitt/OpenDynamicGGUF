"""Step 08 — activation features (forward hooks or proxy)."""

from .features import compute_catalog_activation_features
from .types import ActivationFeaturesResult

__all__ = ["ActivationFeaturesResult", "compute_catalog_activation_features"]
