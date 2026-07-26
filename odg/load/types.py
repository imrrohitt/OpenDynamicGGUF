"""Types for Step 02 (model load)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

LoadBackend = Literal["gguf", "hf_safetensors", "hf_transformers"]


@dataclass
class TensorInfo:
    """Lightweight tensor descriptor (full enumerate is Step 03)."""

    name: str
    shape: list[int]
    dtype: str
    nbytes_approx: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LoadedModel:
    """
    Result of Step 02.

    We do not keep a live PyTorch module in the checkpoint — that cannot be
    serialized. Instead we record everything later steps need to reopen the
    same source (path, backend, metadata, tensor index summary).
    """

    backend: LoadBackend
    source_path: str
    source_is_quantized: bool
    architecture: str | None
    n_tensors: int
    file_size_bytes: int
    parameter_count: int | None
    layer_count: int | None
    embedding_length: int | None
    context_length: int | None
    vocab_size: int | None
    dtype_summary: dict[str, int] = field(default_factory=dict)
    sample_tensors: list[TensorInfo] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    steps_log: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "source_path": self.source_path,
            "source_is_quantized": self.source_is_quantized,
            "architecture": self.architecture,
            "n_tensors": self.n_tensors,
            "file_size_bytes": self.file_size_bytes,
            "parameter_count": self.parameter_count,
            "layer_count": self.layer_count,
            "embedding_length": self.embedding_length,
            "context_length": self.context_length,
            "vocab_size": self.vocab_size,
            "dtype_summary": self.dtype_summary,
            "sample_tensors": [t.to_dict() for t in self.sample_tensors],
            "metadata": self.metadata,
            "steps_log": self.steps_log,
            "notes": self.notes,
        }
