"""Types for Step 09 (freeze GGUF reference)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FreezeResult:
    model_ref: str
    method: str  # "hf_convert_bf16" | "promote_source_gguf"
    gguf_path: str
    gguf_sha256: str
    gguf_nbytes: int
    is_bf16_reference: bool
    source_path: str | None
    source_is_quantized: bool
    dtype_summary: dict[str, int]
    n_tensors: int
    catalog_match: bool
    catalog_missing: list[str]
    steps_log: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary_dict(self) -> dict[str, Any]:
        return asdict(self)
