"""Types for Step 03 (tensor enumeration)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TensorRow:
    index: int
    name: str
    shape: list[int]
    dtype: str
    n_elements: int
    nbytes: int
    layer: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EnumerationResult:
    n_tensors: int
    total_elements: int
    total_nbytes: int
    dtype_summary: dict[str, int]
    layer_summary: dict[str, int]
    prefix_summary: dict[str, int]
    tensors: list[TensorRow]
    steps_log: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_tensors": self.n_tensors,
            "total_elements": self.total_elements,
            "total_nbytes": self.total_nbytes,
            "dtype_summary": self.dtype_summary,
            "layer_summary": self.layer_summary,
            "prefix_summary": self.prefix_summary,
            "tensors": [t.to_dict() for t in self.tensors],
            "steps_log": self.steps_log,
            "notes": self.notes,
        }

    def summary_dict(self) -> dict[str, Any]:
        """Compact output for output.json (full list lives in tensors.json)."""
        return {
            "n_tensors": self.n_tensors,
            "total_elements": self.total_elements,
            "total_nbytes": self.total_nbytes,
            "dtype_summary": self.dtype_summary,
            "layer_summary": self.layer_summary,
            "prefix_summary": self.prefix_summary,
            "sample_tensors": [t.to_dict() for t in self.tensors[:15]],
            "steps_log": self.steps_log,
            "notes": self.notes,
        }
