"""Step 08 — activation features (forward hooks or proxy)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import math
from pathlib import Path
import hashlib
import json
from typing import Any, Literal
from corpus import estimate_tokens


# --- from activation_features/types.py ---
@dataclass
class ActivationFeaturesResult:
    model_ref: str
    method: str  # "forward_hooks" | "proxy_from_weights"
    calib_path: str | None
    n_docs_used: int
    n_tokens_est: int
    n_tensors: int
    n_with_features: int
    catalog_sha256: str
    hardest_groups: list[dict[str, Any]]
    easiest_groups: list[dict[str, Any]]
    steps_log: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary_dict(self) -> dict[str, Any]:
        return asdict(self)

# --- from activation_features/proxy.py ---
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

# --- from activation_features/forward.py ---
def forward_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def run_forward_activation_stats(
    *,
    model_id_or_path: str,
    calib_docs: list[str],
    hf_name_to_gguf: dict[str, str],
    max_docs: int = 32,
    max_seq_len: int = 256,
    device: str | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Run a short calib forward pass; return stats keyed by GGUF tensor name.

    Hooks Linear module *inputs* and maps HF param names → GGUF via catalog.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Map module qualified name → gguf name (best-effort via weight param name)
    module_to_gguf: dict[str, str] = {}
    for full_name, _p in model.named_parameters():
        if not full_name.endswith(".weight"):
            continue
        gguf = hf_name_to_gguf.get(full_name)
        if gguf:
            # Linear module is parent of .weight
            mod_name = full_name[: -len(".weight")]
            module_to_gguf[mod_name] = gguf

    stats: dict[str, dict[str, float]] = {}

    def make_hook(gguf_name: str):
        def hook(_m, inputs, _out):
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            xf = x.detach().float()
            amin = float(xf.min().item())
            amax = float(xf.max().item())
            aabs = float(xf.abs().max().item())
            flat = xf.reshape(-1)
            # outlier vs running std of this batch
            std = float(flat.std(unbiased=False).item()) + 1e-12
            out_r = float((flat.abs() > 6.0 * std).float().mean().item())
            rms = float(torch.sqrt((flat * flat).mean()).item())
            s = stats.setdefault(
                gguf_name,
                {
                    "range_min": amin,
                    "range_max": amax,
                    "absmax": aabs,
                    "outlier_ratio_sum": out_r,
                    "outlier_n": 1.0,
                    "channel_rms_sum": rms,
                    "n_batches": 1.0,
                },
            )
            s["range_min"] = min(s["range_min"], amin)
            s["range_max"] = max(s["range_max"], amax)
            s["absmax"] = max(s["absmax"], aabs)
            s["outlier_ratio_sum"] += out_r
            s["outlier_n"] += 1.0
            s["channel_rms_sum"] += rms
            s["n_batches"] += 1.0

        return hook

    handles = []
    for mod_name, mod in model.named_modules():
        if mod_name in module_to_gguf and hasattr(mod, "forward"):
            # Only hook leaf Linears
            if mod.__class__.__name__ in {"Linear", "Conv1D"}:
                handles.append(
                    mod.register_forward_hook(make_hook(module_to_gguf[mod_name]))
                )

    docs = calib_docs[:max_docs]
    with torch.no_grad():
        for text in docs:
            enc = tok(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_len,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            model(**enc)

    for h in handles:
        h.remove()

    out: dict[str, dict[str, Any]] = {}
    for name, s in stats.items():
        n = max(s["outlier_n"], 1.0)
        out[name] = {
            "range_min": s["range_min"],
            "range_max": s["range_max"],
            "absmax": s["absmax"],
            "outlier_ratio": s["outlier_ratio_sum"] / n,
            "channel_rms_mean": s["channel_rms_sum"] / n,
            "method": "forward_hooks",
            "n_batches": int(s["n_batches"]),
        }
    return out

# --- from activation_features/features.py ---
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
