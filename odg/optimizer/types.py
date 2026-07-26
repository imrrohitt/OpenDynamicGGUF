"""Types for Step 13 (recipe optimizer)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class OptimizeResult:
    model_ref: str
    method: str
    budget_bytes: int
    estimated_bytes: int
    predicted_delta_kld: float
    n_groups: int
    recipe_path: str
    tensor_type_file: str
    pareto_paths: list[str]
    assignments: dict[str, str]
    steps_log: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary_dict(self) -> dict[str, Any]:
        return asdict(self)
