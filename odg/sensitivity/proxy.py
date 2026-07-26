"""
Proxy ΔKLD / Δbytes estimates when llama-quantize + perplexity are unavailable.

Produces the same sensitivity table shape the optimizer expects.
Marked method=proxy_from_features — replace with real probes for production.
"""

from __future__ import annotations

import math
import re
from typing import Any

# Approx bytes per element for common GGUF types (block-aware averages)
BYTES_PER_ELEM: dict[str, float] = {
    "BF16": 2.0,
    "F16": 2.0,
    "F32": 4.0,
    "Q8_0": 34.0 / 32.0,
    "Q6_K": 210.0 / 256.0,
    "Q5_K": 176.0 / 256.0,
    "Q4_K": 144.0 / 256.0,
    "Q3_K": 110.0 / 256.0,
    "Q2_K": 84.0 / 256.0,
}

# Default trial ladder (easy → hard groups try lower first in ranking, not here)
DEFAULT_PROBE_TYPES = ["Q3_K", "Q4_K", "Q5_K", "Q6_K"]
BASELINE_TYPE = "Q6_K"

# Role → base ΔKLD scale at Q4_K (heuristic)
_ROLE_KLD: dict[str, float] = {
    "attn_q": 0.035,
    "attn_k": 0.025,
    "attn_v": 0.045,
    "attn_o": 0.030,
    "ffn_gate": 0.012,
    "ffn_up": 0.010,
    "ffn_down": 0.022,
    "embedding": 0.055,
    "lm_head": 0.040,
    "other": 0.020,
}

_DEPTH_KLD: dict[str, float] = {
    "early": 0.95,
    "middle": 1.00,
    "late": 1.20,
    "global": 1.10,
}

# Quant type → multiplier on base KLD (lower bits → higher KLD)
_QUANT_KLD_MULT: dict[str, float] = {
    "Q2_K": 3.5,
    "Q3_K": 2.2,
    "Q4_K": 1.0,
    "Q5_K": 0.55,
    "Q6_K": 0.25,
    "Q8_0": 0.08,
}


def estimate_group_nbytes(n_elements: int, quant: str) -> int:
    bpe = BYTES_PER_ELEM.get(quant.upper())
    if bpe is None:
        raise KeyError(f"Unknown quant type for size estimate: {quant}")
    return int(math.ceil(n_elements * bpe))


def _group_n_elements(group: dict[str, Any], tensors: dict[str, Any]) -> int:
    total = 0
    for name in group.get("tensor_names") or []:
        t = tensors.get(name) or {}
        total += int(t.get("n_elements") or 0)
    return total


def _feature_hardness(group: dict[str, Any], tensors: dict[str, Any]) -> float:
    """Combine weight + activation + imatrix-like signals into [0, ~2]."""
    names = group.get("tensor_names") or []
    w_outs, a_abs, a_outs = [], [], []
    for n in names:
        t = tensors.get(n) or {}
        wf = t.get("weight_features") or {}
        af = t.get("activation_features") or {}
        if wf.get("outlier_ratio") is not None:
            w_outs.append(float(wf["outlier_ratio"]))
        if af.get("absmax") is not None:
            a_abs.append(float(af["absmax"]))
        if af.get("outlier_ratio") is not None:
            a_outs.append(float(af["outlier_ratio"]))
    # Prefer precomputed group features when present
    gwf = group.get("weight_features") or {}
    gaf = group.get("activation_features") or {}
    w_out = float(gwf.get("outlier_ratio_mean") or (sum(w_outs) / len(w_outs) if w_outs else 0.0))
    a_max = float(gaf.get("absmax") or (max(a_abs) if a_abs else 0.0))
    a_out = float(gaf.get("outlier_ratio") or (sum(a_outs) / len(a_outs) if a_outs else 0.0))
    wh = float(gwf.get("hardness") or 0.0)
    ah = float(gaf.get("hardness") or 0.0)
    return 0.4 * wh + 0.4 * ah + 20.0 * w_out + 0.05 * a_max + 10.0 * a_out


def _proxy_delta_kld(
    group: dict[str, Any],
    tensors: dict[str, Any],
    quant: str,
    *,
    imatrix_group_importance: float | None = None,
) -> float:
    role = str(group.get("role") or "other")
    depth = str(group.get("depth") or "global")
    base = _ROLE_KLD.get(role, 0.02) * _DEPTH_KLD.get(depth, 1.0)
    qmult = _QUANT_KLD_MULT.get(quant.upper(), 1.0)
    hard = _feature_hardness(group, tensors)
    # Normalize hardness roughly into a 0.5–2.0 multiplier
    hard_m = 0.5 + min(hard, 5.0) / 5.0 * 1.5
    imp_m = 1.0
    if imatrix_group_importance is not None:
        imp_m = 0.7 + 0.6 * float(imatrix_group_importance)
    return max(1e-6, base * qmult * hard_m * imp_m)


def tensor_type_regex(group: dict[str, Any]) -> str:
    """
    Build a llama-quantize --tensor-type regex covering the group's tensors.
    Example: blk.(0|1|2).ffn_up.weight → pattern on shared suffix.
    """
    names = group.get("tensor_names") or []
    if not names:
        return re.escape(str(group.get("group_id") or "unknown"))
    # Common case: blk.N.ROLE.weight
    m = re.match(r"blk\.(\d+)\.(.+)$", names[0])
    if m and all(re.match(r"blk\.\d+\." + re.escape(m.group(2)) + r"$", n) for n in names):
        layers = []
        suffix = m.group(2)
        for n in names:
            mm = re.match(r"blk\.(\d+)\.", n)
            if mm:
                layers.append(mm.group(1))
        layers_sorted = sorted(layers, key=int)
        return rf"blk\.({'|'.join(layers_sorted)})\.{re.escape(suffix)}"
    # Fallback: alternation of escaped full names
    return "(?:" + "|".join(re.escape(n) for n in names) + ")"


def probe_groups_proxy(
    catalog: dict[str, Any],
    *,
    probe_types: list[str] | None = None,
    baseline_type: str = BASELINE_TYPE,
    imatrix_groups: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Return sensitivity rows for all quantizable groups × probe types.
    """
    probe_types = probe_types or list(DEFAULT_PROBE_TYPES)
    tensors = catalog.get("tensors") or {}
    groups = catalog.get("groups") or {}
    rows: list[dict[str, Any]] = []

    for gid, g in sorted(groups.items()):
        if not g.get("quantizable", True):
            continue
        n_elem = _group_n_elements(g, tensors)
        if n_elem <= 0:
            continue
        base_bytes = estimate_group_nbytes(n_elem, baseline_type)
        imp = None
        if imatrix_groups and gid in imatrix_groups:
            imp = float(imatrix_groups[gid].get("importance_mean") or 0.0)

        for q in probe_types:
            q_bytes = estimate_group_nbytes(n_elem, q)
            delta_bytes = base_bytes - q_bytes  # positive = smaller
            # If probing higher than baseline, bytes_saved may be negative
            delta_kld = _proxy_delta_kld(g, tensors, q, imatrix_group_importance=imp)
            # If quant is higher precision than baseline, KLD should be near 0 vs baseline
            if BYTES_PER_ELEM.get(q.upper(), 99) >= BYTES_PER_ELEM.get(
                baseline_type.upper(), 1
            ):
                delta_kld *= 0.15
            eps = 1e-6
            score = (delta_bytes / max(delta_kld, eps)) if delta_bytes > 0 else 0.0
            hint = "compress" if score > 5e7 and delta_kld < 0.03 else (
                "pin_high" if delta_kld > 0.04 else "neutral"
            )
            rows.append(
                {
                    "group_id": gid,
                    "role": g.get("role"),
                    "depth": g.get("depth"),
                    "probe": q,
                    "baseline": baseline_type,
                    "n_elements": n_elem,
                    "n_tensors": g.get("n_tensors"),
                    "bytes_baseline": base_bytes,
                    "bytes_probe": q_bytes,
                    "delta_bytes": delta_bytes,
                    "delta_kld": delta_kld,
                    "top_token_agree": max(0.0, 1.0 - 2.5 * delta_kld),
                    "efficiency": score,
                    "decision_hint": hint,
                    "tensor_type_regex": tensor_type_regex(g),
                    "method": "proxy_from_features",
                    "split": "search",
                }
            )
    return rows
