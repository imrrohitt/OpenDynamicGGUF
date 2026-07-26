"""Types for Step 11 (reference logits)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class LogitsResult:
    model_ref: str
    method: str  # "llama_perplexity" | "proxy_manifest"
    gguf_path: str
    gguf_sha256: str | None
    search_path: str
    heldout_path: str
    search_sha256: str | None
    heldout_sha256: str | None
    logits_search_path: str | None
    logits_heldout_path: str | None
    logits_search_sha256: str | None
    logits_heldout_sha256: str | None
    cache_key: str
    steps_log: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary_dict(self) -> dict[str, Any]:
        return asdict(self)
