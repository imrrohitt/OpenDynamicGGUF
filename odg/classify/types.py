"""Types for Step 04 (tensor classification)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ClassifiedTensor:
    index: int
    name: str
    shape: list[int]
    dtype: str
    n_elements: int
    nbytes: int
    role: str
    layer: int | None
    depth: str | None
    group_id: str
    quantizable: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClassificationResult:
    n_tensors: int
    n_layers: int
    role_summary: dict[str, int]
    depth_summary: dict[str, int]
    quantizable_summary: dict[str, int]
    group_summary: dict[str, int]
    other_names: list[str]
    coverage: float
    tensors: list[ClassifiedTensor]
    steps_log: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_tensors": self.n_tensors,
            "n_layers": self.n_layers,
            "role_summary": self.role_summary,
            "depth_summary": self.depth_summary,
            "quantizable_summary": self.quantizable_summary,
            "group_summary": self.group_summary,
            "other_names": self.other_names,
            "coverage": self.coverage,
            "tensors": [t.to_dict() for t in self.tensors],
            "steps_log": self.steps_log,
            "notes": self.notes,
        }

    def summary_dict(self) -> dict[str, Any]:
        return {
            "n_tensors": self.n_tensors,
            "n_layers": self.n_layers,
            "role_summary": self.role_summary,
            "depth_summary": self.depth_summary,
            "quantizable_summary": self.quantizable_summary,
            "group_summary": self.group_summary,
            "other_names": self.other_names,
            "coverage": self.coverage,
            "sample_tensors": [t.to_dict() for t in self.tensors[:20]],
            "steps_log": self.steps_log,
            "notes": self.notes,
        }
