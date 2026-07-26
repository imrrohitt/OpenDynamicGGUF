"""Types for Step 14 (export)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ExportResult:
    model_ref: str
    method: str  # "llama_quantize" | "dry_run"
    gguf_in: str
    gguf_out: str | None
    gguf_out_sha256: str | None
    gguf_out_nbytes: int | None
    recipe_path: str
    tensor_type_file: str
    imatrix_path: str | None
    command: list[str]
    estimated_bytes: int | None
    steps_log: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary_dict(self) -> dict[str, Any]:
        return asdict(self)
