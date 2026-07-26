"""Types for Step 12 (sensitivity probe)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SensitivityResult:
    model_ref: str
    method: str  # "llama_probe" | "proxy_from_features"
    gguf_sha256: str | None
    search_path: str | None
    n_groups_probed: int
    n_rows: int
    probe_types: list[str]
    baseline_type: str
    top_efficiency: list[dict[str, Any]]
    pinned_hints: list[dict[str, Any]]
    steps_log: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary_dict(self) -> dict[str, Any]:
        return asdict(self)
