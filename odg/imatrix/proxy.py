"""
Proxy importance scores when llama-imatrix is unavailable.

Not a drop-in for llama-quantize --imatrix, but preserves per-tensor ranking
for recipe plumbing until a real imatrix.gguf is built.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_proxy_importance(
    catalog: dict[str, Any],
    *,
    out_path: Path,
    gguf_sha256: str | None,
    calib_path: str,
) -> dict[str, Any]:
    """
    Write imatrix_proxy.json with per-tensor importance in [0, 1].
    """
    tensors = catalog.get("tensors") or {}
    scores: dict[str, dict[str, Any]] = {}
    raw: list[tuple[str, float]] = []

    for name, t in tensors.items():
        if not t.get("quantizable", True):
            continue
        wf = t.get("weight_features") or {}
        af = t.get("activation_features") or {}
        w_out = float(wf.get("outlier_ratio") or 0.0)
        w_var = float(wf.get("variance") or 0.0)
        a_abs = float(af.get("absmax") or 0.0)
        a_out = float(af.get("outlier_ratio") or 0.0)
        # Higher = protect more when rounding
        score = (
            2.0 * w_out
            + 0.15 * (w_var ** 0.5)
            + 0.05 * a_abs
            + 1.5 * a_out
        )
        role = str(t.get("role") or "")
        if role in {"attn_q", "attn_k", "attn_v", "attn_o"}:
            score *= 1.15
        if role in {"ffn_gate", "ffn_up"}:
            score *= 1.05
        raw.append((name, score))

    if not raw:
        payload = {
            "method": "proxy_importance",
            "gguf_sha256": gguf_sha256,
            "calib_path": calib_path,
            "n_tensors": 0,
            "tensors": {},
            "note": "No quantizable tensors found in catalog",
        }
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return payload

    mx = max(s for _, s in raw) or 1.0
    for name, score in raw:
        t = tensors[name]
        scores[name] = {
            "importance": float(score / mx),
            "raw_score": float(score),
            "role": t.get("role"),
            "group_id": t.get("group_id"),
            "quantizable": True,
        }

    # Group aggregates
    groups: dict[str, list[float]] = {}
    for name, info in scores.items():
        gid = str(info.get("group_id") or "unknown")
        groups.setdefault(gid, []).append(float(info["importance"]))
    group_scores = {
        gid: {
            "importance_mean": sum(v) / len(v),
            "importance_max": max(v),
            "n_tensors": len(v),
        }
        for gid, v in groups.items()
    }

    ranked = sorted(
        ({"name": n, **info} for n, info in scores.items()),
        key=lambda r: r["importance"],
        reverse=True,
    )

    payload = {
        "method": "proxy_importance",
        "gguf_sha256": gguf_sha256,
        "calib_path": calib_path,
        "n_tensors": len(scores),
        "tensors": scores,
        "groups": group_scores,
        "top_important": ranked[:15],
        "least_important": list(reversed(ranked[-10:])),
        "note": (
            "Proxy only — not usable as llama-quantize --imatrix. "
            "Install llama-imatrix and re-run with --mode llama."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
