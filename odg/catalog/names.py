"""GGUF ↔ Hugging Face tensor name mapping (Gemma / Llama-style)."""

from __future__ import annotations

import re

# GGUF blk.N.<suffix> → HF layers.N.<path>
_GGUF_BLK_TO_HF: dict[str, str] = {
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "post_attention_norm.weight": "post_attention_norm.weight",
    "post_ffw_norm.weight": "post_feedforward_layernorm.weight",
}

_GGUF_GLOBAL_TO_HF: dict[str, str] = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "output_norm.weight": "model.norm.weight",
}

_BLK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")


def looks_like_gguf_name(name: str) -> bool:
    return name.startswith("blk.") or name in _GGUF_GLOBAL_TO_HF or name.startswith("token_embd")


def gguf_to_hf(gguf_name: str) -> str | None:
    if gguf_name in _GGUF_GLOBAL_TO_HF:
        return _GGUF_GLOBAL_TO_HF[gguf_name]
    m = _BLK_RE.match(gguf_name)
    if not m:
        return None
    layer, suffix = m.group(1), m.group(2)
    hf_suffix = _GGUF_BLK_TO_HF.get(suffix)
    if not hf_suffix:
        return None
    return f"model.layers.{layer}.{hf_suffix}"


def hf_to_gguf(hf_name: str) -> str | None:
    # reverse of above for when source is HF
    for g, h in _GGUF_GLOBAL_TO_HF.items():
        if hf_name == h or hf_name.endswith(h.split(".", 1)[-1] if h.startswith("model.") else h):
            if hf_name.endswith("embed_tokens.weight") or "embed_tokens" in hf_name:
                return "token_embd.weight"
            if hf_name.endswith("lm_head.weight") or hf_name == "lm_head.weight":
                return "output.weight"
            if hf_name.endswith("model.norm.weight") or hf_name.endswith(".norm.weight") and "layers" not in hf_name:
                if "layers" not in hf_name:
                    return "output_norm.weight"

    m = re.search(r"layers?[.\[](\d+)[.\]]?(.*)$", hf_name)
    if not m:
        if "embed_tokens" in hf_name:
            return "token_embd.weight"
        if "lm_head" in hf_name:
            return "output.weight"
        return None
    layer = m.group(1)
    rest = m.group(2).lstrip(".")
    # normalize
    mapping = {
        "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.k_proj.weight": "attn_k.weight",
        "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.o_proj.weight": "attn_output.weight",
        "self_attn.q_norm.weight": "attn_q_norm.weight",
        "self_attn.k_norm.weight": "attn_k_norm.weight",
        "input_layernorm.weight": "attn_norm.weight",
        "mlp.gate_proj.weight": "ffn_gate.weight",
        "mlp.up_proj.weight": "ffn_up.weight",
        "mlp.down_proj.weight": "ffn_down.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
        "post_attention_norm.weight": "post_attention_norm.weight",
        "post_feedforward_layernorm.weight": "post_ffw_norm.weight",
    }
    suffix = mapping.get(rest)
    if suffix:
        return f"blk.{layer}.{suffix}"
    return None
