"""
Step 12 — Sensitivity probing (proxy table; llama path when tools exist).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from .proxy import BASELINE_TYPE, DEFAULT_PROBE_TYPES, probe_groups_proxy
from .types import SensitivityResult

Mode = Literal["auto", "llama", "proxy"]


def _load_imatrix_groups(proxy_path: Path | None) -> dict[str, Any] | None:
    if not proxy_path or not proxy_path.is_file():
        return None
    data = json.loads(proxy_path.read_text())
    return data.get("groups")


def build_sensitivity_table(
    *,
    model_ref: str,
    out_dir: Path,
    catalog: dict[str, Any],
    gguf_sha256: str | None = None,
    search_path: str | Path | None = None,
    imatrix_proxy_path: str | Path | None = None,
    mode: Mode = "auto",
    probe_types: list[str] | None = None,
    baseline_type: str = BASELINE_TYPE,
) -> tuple[SensitivityResult, list[dict[str, Any]]]:
    """
    Write sensitivity.json (+ summary). Returns (result, rows).
    """
    log: list[str] = []
    notes: list[str] = []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    probe_types = probe_types or list(DEFAULT_PROBE_TYPES)
    log.append(f"1. Probe types={probe_types} baseline={baseline_type}")
    log.append("2. Split=search only (heldout forbidden)")

    # Real llama probing is not implemented without binaries + logits caches;
    # auto falls back to proxy (same as previous steps).
    if mode == "llama":
        raise RuntimeError(
            "llama probe mode requires llama-quantize + llama-perplexity + "
            "logits-search.bin. Install llama.cpp, run reference-logits --mode llama, "
            "then re-try. For now use --mode proxy or auto."
        )

    method = "proxy_from_features"
    if mode == "auto":
        log.append(
            "3. auto: llama probe tools/caches not wired — proxy_from_features"
        )
    else:
        log.append("3. mode=proxy — estimating ΔKLD/Δbytes from features")

    imatrix_groups = _load_imatrix_groups(
        Path(imatrix_proxy_path) if imatrix_proxy_path else None
    )
    if imatrix_groups:
        log.append(f"4. Loaded imatrix group importance ({len(imatrix_groups)} groups)")
    else:
        log.append("4. No imatrix proxy groups — features only")

    rows = probe_groups_proxy(
        catalog,
        probe_types=probe_types,
        baseline_type=baseline_type,
        imatrix_groups=imatrix_groups,
    )
    groups_probed = len({r["group_id"] for r in rows})
    log.append(f"5. Probed groups={groups_probed} rows={len(rows)}")

    # Rank by efficiency (bytes saved per unit KLD)
    by_eff = sorted(rows, key=lambda r: r["efficiency"], reverse=True)
    top_efficiency = by_eff[:10]

    # Pin hints: high ΔKLD at Q4_K
    pinned = [
        r
        for r in rows
        if r["probe"] == "Q4_K" and r["decision_hint"] == "pin_high"
    ]
    pinned = sorted(pinned, key=lambda r: r["delta_kld"], reverse=True)[:10]

    notes.append(
        "proxy_from_features estimates ΔKLD — not measured. "
        "Production needs llama-quantize trial + perplexity --kl-divergence on search."
    )
    notes.append("Held-out must not be used in this step.")

    table = {
        "model_ref": model_ref,
        "method": method,
        "gguf_sha256": gguf_sha256,
        "search_path": str(search_path) if search_path else None,
        "baseline_type": baseline_type,
        "probe_types": probe_types,
        "n_groups_probed": groups_probed,
        "n_rows": len(rows),
        "rows": rows,
        "top_efficiency": top_efficiency,
        "pinned_hints": pinned,
        "notes": notes,
    }
    (out_dir / "sensitivity.json").write_text(
        json.dumps(table, indent=2) + "\n", encoding="utf-8"
    )
    log.append("6. Wrote sensitivity.json")

    result = SensitivityResult(
        model_ref=model_ref,
        method=method,
        gguf_sha256=gguf_sha256,
        search_path=str(search_path) if search_path else None,
        n_groups_probed=groups_probed,
        n_rows=len(rows),
        probe_types=probe_types,
        baseline_type=baseline_type,
        top_efficiency=[
            {
                "group_id": r["group_id"],
                "probe": r["probe"],
                "delta_bytes": r["delta_bytes"],
                "delta_kld": r["delta_kld"],
                "efficiency": r["efficiency"],
                "decision_hint": r["decision_hint"],
            }
            for r in top_efficiency
        ],
        pinned_hints=[
            {
                "group_id": r["group_id"],
                "probe": r["probe"],
                "delta_kld": r["delta_kld"],
                "decision_hint": r["decision_hint"],
            }
            for r in pinned
        ],
        steps_log=log,
        notes=notes,
    )
    return result, rows
