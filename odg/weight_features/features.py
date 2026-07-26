"""
Step 06 — Compute weight features from tensors alone (no calibration text).

Features prioritize probe order; they do NOT decide final bit widths.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from odg.gguf_tensors import gguf_tensor_map, read_tensor_f32

from .types import WeightFeaturesResult

EPS = 1e-12


def compute_weight_features(
    w: np.ndarray,
    *,
    shape: list[int] | None = None,
    spectral: bool = True,
    max_spectral_dim: int = 4096,
) -> dict[str, Any]:
    """Stats for one weight tensor (flat float32 array)."""
    x = np.asarray(w, dtype=np.float32).reshape(-1)
    n = int(x.size)
    if n == 0:
        return {
            "mean": 0.0,
            "variance": 0.0,
            "sparsity": 0.0,
            "outlier_ratio": 0.0,
            "weight_norm": 0.0,
            "entropy": 0.0,
            "spectral_norm": None,
            "n_elements": 0,
        }

    mean = float(x.mean())
    var = float(x.var())  # population
    std = math.sqrt(var) + EPS
    abs_x = np.abs(x)
    sparsity = float((abs_x < 1e-3).mean())
    outlier_ratio = float((abs_x > 6.0 * std).mean())
    weight_norm = float(np.linalg.norm(x))

    # Histogram entropy (bits) — cheap proxy for "how peaked" the distribution is
    hist, _ = np.histogram(x, bins=64)
    p = hist.astype(np.float64)
    p = p[p > 0]
    p /= p.sum()
    entropy = float(-(p * np.log2(p)).sum())

    spectral_norm = None
    if spectral and shape is not None and len(shape) == 2:
        rows, cols = int(shape[0]), int(shape[1])
        if rows * cols == n and max(rows, cols) <= max_spectral_dim:
            spectral_norm = _approx_spectral_norm(x.reshape(rows, cols))

    return {
        "mean": mean,
        "variance": var,
        "sparsity": sparsity,
        "outlier_ratio": outlier_ratio,
        "weight_norm": weight_norm,
        "entropy": entropy,
        "spectral_norm": spectral_norm,
        "n_elements": n,
    }


def _approx_spectral_norm(mat: np.ndarray, iters: int = 8) -> float:
    """Power iteration ≈ largest singular value."""
    a = mat.astype(np.float32, copy=False)
    m, n = a.shape
    v = np.ones(n, dtype=np.float32) / math.sqrt(n)
    for _ in range(iters):
        u = a @ v
        un = float(np.linalg.norm(u)) + EPS
        u /= un
        v = a.T @ u
        vn = float(np.linalg.norm(v)) + EPS
        v /= vn
    return float(np.linalg.norm(a @ v))


def _hardness(feats: dict[str, Any]) -> float:
    """Heuristic ranking score (higher = harder to quantize)."""
    outlier = float(feats.get("outlier_ratio") or 0.0)
    var = float(feats.get("variance") or 0.0)
    sparsity = float(feats.get("sparsity") or 0.0)
    spectral = feats.get("spectral_norm")
    spectral_f = float(spectral) if spectral is not None else 0.0
    # outliers + variance + spectral hurt; sparsity helps (easy)
    return outlier * 100.0 + math.sqrt(max(var, 0.0)) + 0.1 * spectral_f - 0.5 * sparsity


def aggregate_group_features(
    member_feats: list[dict[str, Any]],
) -> dict[str, Any]:
    if not member_feats:
        return {}
    keys = ["mean", "variance", "sparsity", "outlier_ratio", "weight_norm", "entropy"]
    out: dict[str, Any] = {"n_tensors": len(member_feats)}
    for k in keys:
        vals = [float(f[k]) for f in member_feats if f.get(k) is not None]
        if not vals:
            continue
        out[f"{k}_mean"] = float(sum(vals) / len(vals))
        out[f"{k}_max"] = float(max(vals))
    specs = [float(f["spectral_norm"]) for f in member_feats if f.get("spectral_norm") is not None]
    if specs:
        out["spectral_norm_mean"] = float(sum(specs) / len(specs))
        out["spectral_norm_max"] = float(max(specs))
    out["hardness"] = _hardness(
        {
            "outlier_ratio": out.get("outlier_ratio_mean", 0.0),
            "variance": out.get("variance_mean", 0.0),
            "sparsity": out.get("sparsity_mean", 0.0),
            "spectral_norm": out.get("spectral_norm_mean"),
        }
    )
    return out


def _catalog_sha256(catalog: dict[str, Any]) -> str:
    """Hash catalog body excluding the sha field itself."""
    body = {k: v for k, v in catalog.items() if k != "catalog_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def compute_catalog_weight_features(
    catalog: dict[str, Any],
    *,
    source_path: str | Path | None = None,
    only_quantizable: bool = False,
) -> tuple[dict[str, Any], WeightFeaturesResult]:
    """
    Fill weight_features on catalog tensors + group aggregates.

    Returns (updated_catalog_dict, summary_result).
    """
    log: list[str] = []
    path = Path(source_path or catalog.get("source_path") or "")
    if not path.is_file():
        raise FileNotFoundError(f"GGUF source not found: {path}")

    log.append(f"1. Opening GGUF for weight reads: {path}")
    gmap = gguf_tensor_map(path)
    data_offset = gmap["data_offset"]
    index = gmap["tensors"]
    log.append(f"2. Tensor index ready (data_offset={data_offset}, n={len(index)})")

    tensors = catalog.get("tensors") or {}
    groups = catalog.get("groups") or {}
    n_with = 0
    n_skip = 0
    skipped: list[str] = []

    for name, t in tensors.items():
        if only_quantizable and not t.get("quantizable", True):
            n_skip += 1
            continue
        info = index.get(name)
        if info is None:
            n_skip += 1
            skipped.append(name)
            t["weight_features"] = None
            continue
        try:
            arr = read_tensor_f32(path, info, data_offset)
            feats = compute_weight_features(arr, shape=list(t.get("shape") or info["shape"]))
            feats["dtype_source"] = info["dtype"]
            feats["from_quantized_source"] = bool(catalog.get("source_is_quantized"))
            t["weight_features"] = feats
            n_with += 1
        except Exception as exc:  # noqa: BLE001
            n_skip += 1
            skipped.append(f"{name}: {exc}")
            t["weight_features"] = None

    log.append(f"3. Per-tensor features: filled={n_with} skipped={n_skip}")

    group_features: dict[str, dict[str, Any]] = {}
    for gid, g in groups.items():
        names = g.get("tensor_names") or []
        member = []
        for n in names:
            tf = (tensors.get(n) or {}).get("weight_features")
            if tf:
                member.append(tf)
        gf = aggregate_group_features(member)
        group_features[gid] = gf
        g["weight_features"] = gf

    # Rank only quantizable groups — norms inflate hardness but are never probed.
    ranked = sorted(
        (
            {"group_id": gid, **gf}
            for gid, gf in group_features.items()
            if gf.get("n_tensors")
            and (groups.get(gid) or {}).get("quantizable", True)
        ),
        key=lambda r: r.get("hardness", 0.0),
        reverse=True,
    )
    hardest = ranked[:5]
    easiest = list(reversed(ranked[-5:])) if ranked else []

    catalog["group_features"] = group_features
    catalog["catalog_sha256"] = _catalog_sha256(catalog)
    notes = [
        "weight_features prioritize probe order; ΔKLD (Step 12) decides bits.",
    ]
    if catalog.get("source_is_quantized"):
        notes.append(
            "Source is already quantized (e.g. Q8_0). Features are from dequantized "
            "weights — useful for plumbing; prefer BF16/HF for production quality ranking."
        )
    if skipped:
        notes.append(f"Skipped {len(skipped)} tensor(s); see log.")
        log.append("4. Skips: " + "; ".join(skipped[:8]))
    else:
        log.append("4. All requested tensors got weight_features")

    log.append(f"5. catalog_sha256={catalog['catalog_sha256'][:16]}…")
    log.append("6. Group hardness ranking ready (hardest first for probe order)")

    result = WeightFeaturesResult(
        model_ref=str(catalog.get("model_ref") or ""),
        source_path=str(path),
        source_is_quantized=bool(catalog.get("source_is_quantized")),
        n_tensors=len(tensors),
        n_with_features=n_with,
        n_skipped=n_skip,
        catalog_sha256=catalog["catalog_sha256"],
        group_features=group_features,
        hardest_groups=hardest,
        easiest_groups=easiest,
        steps_log=log,
        notes=notes,
    )
    return catalog, result
