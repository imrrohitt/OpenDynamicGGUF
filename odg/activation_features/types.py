"""Types for Step 08 (activation features)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ActivationFeaturesResult:
    model_ref: str
    method: str  # "forward_hooks" | "proxy_from_weights"
    calib_path: str | None
    n_docs_used: int
    n_tokens_est: int
    n_tensors: int
    n_with_features: int
    catalog_sha256: str
    hardest_groups: list[dict[str, Any]]
    easiest_groups: list[dict[str, Any]]
    steps_log: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary_dict(self) -> dict[str, Any]:
        return asdict(self)
