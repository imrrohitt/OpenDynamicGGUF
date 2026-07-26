"""
Optional BF16 / HF forward-pass activation hooks (requires torch + transformers).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


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
