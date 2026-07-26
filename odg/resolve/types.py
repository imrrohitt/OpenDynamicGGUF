"""Shared types for Step 01 (model resolution)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class SourceKind(str, Enum):
    """How the user named the model."""

    HF = "hf"
    OLLAMA = "ollama"
    MLX = "mlx"
    LOCAL = "local"


@dataclass
class ArchitectureDescriptor:
    """What we know about the model after resolving (before full load)."""

    family: str | None = None
    layer_count: int | None = None
    embedding_length: int | None = None
    parameter_count: int | None = None
    context_length: int | None = None
    is_moe: bool = False
    is_hybrid_ssm: bool = False
    chat_template: str | None = None
    specialty_domain: str | None = None
    # Provenance / warnings
    ollama_quantization: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResolvedModel:
    """
    Result of Step 01.

    For Ollama (default right now):
      - ``local_path`` is the local Ollama GGUF blob
      - ``source_is_quantized`` is True if that blob is Q4/Q8/…
      - ``hf_repo_id`` is still recorded as the ideal BF16 upstream for later

    For HF / local BF16:
      - ``local_path`` points at full-precision weights when available
    """

    user_ref: str
    kind: SourceKind
    hf_repo_id: str | None
    local_path: str | None
    weights_ready: bool
    source_sha256: str | None
    descriptor: ArchitectureDescriptor
    source_is_quantized: bool = False
    rejected_quantized_source: str | None = None
    steps_log: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_ref": self.user_ref,
            "kind": self.kind.value,
            "hf_repo_id": self.hf_repo_id,
            "local_path": self.local_path,
            "weights_ready": self.weights_ready,
            "source_sha256": self.source_sha256,
            "source_is_quantized": self.source_is_quantized,
            "rejected_quantized_source": self.rejected_quantized_source,
            "descriptor": self.descriptor.to_dict(),
            "steps_log": self.steps_log,
        }
