"""Types for Step 07 (calibration corpus)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CorpusResult:
    model_ref: str
    corpus_id: str
    chat_template: str | None
    specialty_domain: str | None
    seed: int
    target_tokens: int
    splits: dict[str, float]
    n_documents: int
    n_calib: int
    n_search: int
    n_heldout: int
    tokens_est_total: int
    tokens_est_calib: int
    tokens_est_search: int
    tokens_est_heldout: int
    chars_total: int
    domain_counts: dict[str, int]
    files: dict[str, str]
    steps_log: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary_dict(self) -> dict[str, Any]:
        return asdict(self)
