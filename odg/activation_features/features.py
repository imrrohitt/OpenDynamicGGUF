"""
Step 08 — Activation features from calib forward pass (or proxy fallback).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from odg.corpus.build import estimate_tokens

from .forward import forward_available, run_forward_activation_stats
from .proxy import aggregate_activation_group, proxy_activation_features
from .types import ActivationFeaturesResult

Mode = Literal["auto", "forward", "proxy"]


def _catalog_sha256(catalog: dict[str, Any]) -> str:
    body = {k: v for k, v in catalog.items() if k != "catalog_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _load_calib_docs(calib_path: Path, *, max_chars: int = 200_000) -> list[str]:
    text = calib_path.read_text(encoding="utf-8")
    # Documents separated by blank lines (as written by Step 07)
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    if not chunks:
        chunks = [text.strip()] if text.strip() else []
    # Cap volume for forward pass memory
    out: list[str] = []
    total = 0
    for c in chunks:
        out.append(c)
        total += len(c)
        if total >= max_chars:
            break
    return out


def _corpus_signal(domain_counts: dict[str, int] | None) -> float:
    """Slight boost when specialty/code heavy (harder activation tails)."""
    if not domain_counts:
        return 1.0
    total = sum(domain_counts.values()) or 1
    hard = domain_counts.get("code", 0) + domain_counts.get("domain", 0)
    return 1.0 + 0.25 * (hard / total)


def compute_catalog_activation_features(
    catalog: dict[str, Any],
    *,
    calib_path: str | Path,
    mode: Mode = "auto",
    hf_model_id: str | None = None,
    hf_local_path: str | None = None,
    max_forward_docs: int = 32,
    corpus_domain_counts: dict[str, int] | None = None,
) -> tuple[dict[str, Any], ActivationFeaturesResult]:
    log: list[str] = []
    calib_path = Path(calib_path)
    if not calib_path.is_file():
        raise FileNotFoundError(f"calib.txt not found: {calib_path}")

    docs = _load_calib_docs(calib_path)
    tokens_est = estimate_tokens(calib_path.read_text(encoding="utf-8"))
    log.append(f"1. Loaded calib: {calib_path} docs={len(docs)} tokens_est≈{tokens_est}")

    method: str
    forward_stats: dict[str, dict[str, Any]] = {}

    want_forward = mode in {"auto", "forward"}
    can_forward = forward_available()
    model_src = hf_local_path or hf_model_id

    if want_forward and can_forward and model_src:
        log.append(f"2. Attempting forward_hooks on {model_src!r}")
        try:
            hf_to_gguf = {}
            for name, t in (catalog.get("tensors") or {}).items():
                hf = t.get("hf_name")
                if hf:
                    hf_to_gguf[hf] = name
            forward_stats = run_forward_activation_stats(
                model_id_or_path=model_src,
                calib_docs=docs,
                hf_name_to_gguf=hf_to_gguf,
                max_docs=max_forward_docs,
            )
            method = "forward_hooks"
            log.append(f"3. Forward pass ok — hooked tensors={len(forward_stats)}")
        except Exception as exc:  # noqa: BLE001
            if mode == "forward":
                raise
            log.append(f"3. Forward failed ({exc}); falling back to proxy_from_weights")
            method = "proxy_from_weights"
            forward_stats = {}
    elif mode == "forward":
        reasons = []
        if not can_forward:
            reasons.append("torch/transformers not installed")
        if not model_src:
            reasons.append("no HF model id/path (need BF16 source)")
        raise RuntimeError(
            "Forward activation features unavailable: " + "; ".join(reasons)
        )
    else:
        method = "proxy_from_weights"
        if want_forward and not can_forward:
            log.append("2. torch/transformers missing — using proxy_from_weights")
        elif want_forward and not model_src:
            log.append(
                "2. No BF16 HF path (Ollama/Q8 source) — using proxy_from_weights"
            )
        else:
            log.append("2. mode=proxy — using proxy_from_weights")

    signal = _corpus_signal(corpus_domain_counts)
    tensors = catalog.get("tensors") or {}
    groups = catalog.get("groups") or {}
    n_with = 0

    for name, t in tensors.items():
        if name in forward_stats:
            feats = forward_stats[name]
        elif method == "forward_hooks" and t.get("quantizable"):
            # Forward ran but this tensor wasn't hooked — light proxy fill
            feats = proxy_activation_features(t, corpus_signal=signal)
            feats["method"] = "proxy_from_weights"
            feats["note"] = "not hooked in forward pass"
        else:
            feats = proxy_activation_features(t, corpus_signal=signal)
        t["activation_features"] = feats
        n_with += 1

    log.append(f"4. activation_features filled on {n_with}/{len(tensors)} tensors")

    group_act: dict[str, dict[str, Any]] = {}
    for gid, g in groups.items():
        members = []
        for n in g.get("tensor_names") or []:
            af = (tensors.get(n) or {}).get("activation_features")
            if af:
                members.append(af)
        gf = aggregate_activation_group(members)
        group_act[gid] = gf
        # Merge into group record without wiping weight_features
        existing = g.get("weight_features")
        g["activation_features"] = gf
        if existing is not None:
            g["weight_features"] = existing

    catalog["group_activation_features"] = group_act
    catalog["catalog_sha256"] = _catalog_sha256(catalog)

    ranked = sorted(
        (
            {"group_id": gid, **gf}
            for gid, gf in group_act.items()
            if gf.get("n_tensors")
            and (groups.get(gid) or {}).get("quantizable", True)
        ),
        key=lambda r: r.get("hardness", 0.0),
        reverse=True,
    )
    hardest = ranked[:5]
    easiest = list(reversed(ranked[-5:])) if ranked else []

    notes = [
        "Activation features prioritize probe order; ΔKLD (Step 12) decides bits.",
    ]
    if method == "proxy_from_weights":
        notes.append(
            "Used proxy_from_weights (no BF16 forward). "
            "Install torch+transformers and use HF BF16 with --mode forward for real hooks."
        )
    else:
        notes.append(
            f"Forward hooks used on up to {max_forward_docs} calib docs "
            "(production: raise --max-docs)."
        )

    log.append(f"5. method={method} catalog_sha256={catalog['catalog_sha256'][:16]}…")
    log.append("6. Group activation hardness ranking ready")

    result = ActivationFeaturesResult(
        model_ref=str(catalog.get("model_ref") or ""),
        method=method,
        calib_path=str(calib_path),
        n_docs_used=min(len(docs), max_forward_docs)
        if method == "forward_hooks"
        else len(docs),
        n_tokens_est=tokens_est,
        n_tensors=len(tensors),
        n_with_features=n_with,
        catalog_sha256=catalog["catalog_sha256"],
        hardest_groups=hardest,
        easiest_groups=easiest,
        steps_log=log,
        notes=notes,
    )
    return catalog, result
