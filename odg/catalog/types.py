"""Types for Step 05 (tensor catalog)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CatalogTensor:
    name: str  # primary key — GGUF name when source is GGUF, else HF
    gguf_name: str | None
    hf_name: str | None
    shape: list[int]
    dtype: str
    role: str
    layer: int | None
    depth: str | None
    group_id: str
    nbytes: int
    n_elements: int
    quantizable: bool
    weight_features: dict[str, Any] | None = None
    activation_features: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CatalogGroup:
    group_id: str
    role: str
    depth: str | None
    quantizable: bool
    n_tensors: int
    total_nbytes: int
    tensor_names: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Catalog:
    model_ref: str
    hf_repo_id: str | None
    source_path: str | None
    source_backend: str | None
    source_is_quantized: bool
    source_sha256: str | None
    catalog_sha256: str
    n_layers: int
    n_tensors: int
    n_groups: int
    n_quantizable: int
    tensors: dict[str, CatalogTensor]
    groups: dict[str, CatalogGroup]
    steps_log: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_ref": self.model_ref,
            "hf_repo_id": self.hf_repo_id,
            "source_path": self.source_path,
            "source_backend": self.source_backend,
            "source_is_quantized": self.source_is_quantized,
            "source_sha256": self.source_sha256,
            "catalog_sha256": self.catalog_sha256,
            "n_layers": self.n_layers,
            "n_tensors": self.n_tensors,
            "n_groups": self.n_groups,
            "n_quantizable": self.n_quantizable,
            "tensors": {k: v.to_dict() for k, v in self.tensors.items()},
            "groups": {k: v.to_dict() for k, v in self.groups.items()},
            "steps_log": self.steps_log,
            "notes": self.notes,
        }

    def summary_dict(self) -> dict[str, Any]:
        return {
            "model_ref": self.model_ref,
            "hf_repo_id": self.hf_repo_id,
            "source_path": self.source_path,
            "source_backend": self.source_backend,
            "source_is_quantized": self.source_is_quantized,
            "source_sha256": self.source_sha256,
            "catalog_sha256": self.catalog_sha256,
            "n_layers": self.n_layers,
            "n_tensors": self.n_tensors,
            "n_groups": self.n_groups,
            "n_quantizable": self.n_quantizable,
            "group_ids": list(self.groups.keys()),
            "sample_tensors": {
                k: v.to_dict()
                for i, (k, v) in enumerate(self.tensors.items())
                if i < 8
            },
            "steps_log": self.steps_log,
            "notes": self.notes,
        }
