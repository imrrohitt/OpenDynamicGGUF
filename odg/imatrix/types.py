"""Types for Step 10 (imatrix)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ImatrixResult:
    model_ref: str
    method: str  # "llama_imatrix" | "proxy_importance"
    gguf_path: str
    gguf_sha256: str | None
    calib_path: str
    imatrix_path: str | None
    imatrix_sha256: str | None
    proxy_path: str | None
    n_chunks: int | None
    n_tensors_scored: int
    steps_log: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary_dict(self) -> dict[str, Any]:
        return asdict(self)
