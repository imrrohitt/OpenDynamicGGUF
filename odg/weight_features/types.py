"""Types for Step 06 (weight features)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class WeightFeaturesResult:
    model_ref: str
    source_path: str | None
    source_is_quantized: bool
    n_tensors: int
    n_with_features: int
    n_skipped: int
    catalog_sha256: str
    group_features: dict[str, dict[str, Any]]
    hardest_groups: list[dict[str, Any]]
    easiest_groups: list[dict[str, Any]]
    steps_log: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary_dict(self) -> dict[str, Any]:
        return asdict(self)
