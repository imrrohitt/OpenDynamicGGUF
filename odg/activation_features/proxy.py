"""
Proxy activation features when a BF16 forward pass is unavailable.

Uses weight_features + role/depth priors. Marked method=proxy_from_weights.
Good for pipeline plumbing; replace with forward_hooks for production ranking.
"""

from __future__ import annotations

import math
from typing import Any

# Roles that tend to show larger activation spikes on real workloads
_ROLE_SCALE: dict[str, float] = {
    "attn_q": 1.35,
    "attn_k": 1.15,
    "attn_v": 1.25,
    "attn_o": 1.20,
    "ffn_gate": 1.40,
    "ffn_up": 1.30,
    "ffn_down": 1.10,
    "embedding": 0.90,
    "lm_head": 1.05,
    "norm": 0.50,
    "other": 1.00,
}

_DEPTH_SCALE: dict[str, float] = {
    "early": 0.95,
    "middle": 1.00,
    "late": 1.15,
    "global": 1.05,
}


def proxy_activation_features(
    tensor: dict[str, Any],
    *,
    corpus_signal: float = 1.0,
) -> dict[str, Any]:
    """
    Estimate activation stats for one catalog tensor from its weight_features.
    """
    wf = tensor.get("weight_features") or {}
    role = str(tensor.get("role") or "other")
    depth = str(tensor.get("depth") or "global")
    n = int(wf.get("n_elements") or tensor.get("n_elements") or 1)

    w_var = float(wf.get("variance") or 0.0)
    w_out = float(wf.get("outlier_ratio") or 0.0)
    w_norm = float(wf.get("weight_norm") or 0.0)
    spectral = wf.get("spectral_norm")
    spectral_f = float(spectral) if spectral is not None else math.sqrt(max(w_var, 0.0)) * 4.0

    role_s = _ROLE_SCALE.get(role, 1.0)
    depth_s = _DEPTH_SCALE.get(depth, 1.0)
    scale = role_s * depth_s * max(corpus_signal, 0.5)

    # Typical activation magnitude proxy: RMS of weights * scale
    rms = math.sqrt(max(w_var, 0.0)) + 1e-12
    absmax = max(rms * 8.0 * scale, spectral_f * 0.5 * scale, 1e-6)
    # Asymmetric-ish range common in post-GELU / residual paths
    range_min = -0.6 * absmax
    range_max = absmax

    # Outliers: weight outliers amplified for sensitive roles
    outlier_ratio = min(0.05, w_out * (1.2 + 0.3 * role_s) * corpus_signal)
    # Channel RMS proxy from weight norm
    channel_rms = (w_norm / math.sqrt(max(n, 1))) * scale

    return {
        "range_min": float(range_min),
        "range_max": float(range_max),
        "absmax": float(absmax),
        "outlier_ratio": float(outlier_ratio),
        "channel_rms_mean": float(channel_rms),
        "method": "proxy_from_weights",
        "role_scale": role_s,
        "depth_scale": depth_s,
    }


def activation_hardness(feats: dict[str, Any]) -> float:
    absmax = float(feats.get("absmax") or 0.0)
    outlier = float(feats.get("outlier_ratio") or 0.0)
    span = float(feats.get("range_max") or 0.0) - float(feats.get("range_min") or 0.0)
    return outlier * 50.0 + 0.1 * absmax + 0.05 * span


def aggregate_activation_group(
    member_feats: list[dict[str, Any]],
) -> dict[str, Any]:
    if not member_feats:
        return {}
    keys = ["range_min", "range_max", "absmax", "outlier_ratio", "channel_rms_mean"]
    out: dict[str, Any] = {"n_tensors": len(member_feats)}
    for k in keys:
        vals = [float(f[k]) for f in member_feats if f.get(k) is not None]
        if not vals:
            continue
        if k == "range_min":
            out[k] = float(min(vals))
        elif k in {"range_max", "absmax"}:
            out[f"{k}_max"] = float(max(vals))
            out[f"{k}_mean"] = float(sum(vals) / len(vals))
        else:
            out[f"{k}_mean"] = float(sum(vals) / len(vals))
            out[f"{k}_max"] = float(max(vals))
    # Unified fields for ranking
    out["absmax"] = out.get("absmax_max", 0.0)
    out["outlier_ratio"] = out.get("outlier_ratio_mean", 0.0)
    out["range_min"] = out.get("range_min", 0.0)
    out["range_max"] = out.get("range_max_max", out.get("range_max_mean", 0.0))
    out["hardness"] = activation_hardness(out)
    methods = {f.get("method") for f in member_feats}
    out["method"] = "forward_hooks" if methods == {"forward_hooks"} else (
        "proxy_from_weights" if methods == {"proxy_from_weights"} else "mixed"
    )
    return out
