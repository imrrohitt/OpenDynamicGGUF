"""Types for Step 15 (validate & release)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ValidateResult:
    model_ref: str
    method: str
    verdict: str  # "RELEASE" | "FAIL" | "PROVISIONAL"
    tier1: dict[str, Any]
    tier2: dict[str, Any]
    tier3: dict[str, Any]
    feedback: list[dict[str, Any]]
    report_path: str
    release_dir: str | None
    steps_log: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary_dict(self) -> dict[str, Any]:
        return asdict(self)
