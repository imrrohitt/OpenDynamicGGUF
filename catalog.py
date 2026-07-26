"""Step 05 — build durable tensor_catalog.json."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import re
import hashlib
import json
from collections import defaultdict


# --- from catalog/types.py ---
@dataclass
class CatalogTensor:
    name: str  # primary key — GGUF name when source is GGUF, else HF
    gguf_name: str | None
    hf_name: str | None
    shape: list[int]
    dtype: str
    role: str
    layer: int | None
    depth: str | None
    group_id: str
    nbytes: int
    n_elements: int
    quantizable: bool
    weight_features: dict[str, Any] | None = None
    activation_features: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CatalogGroup:
    group_id: str
    role: str
    depth: str | None
    quantizable: bool
    n_tensors: int
    total_nbytes: int
    tensor_names: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Catalog:
    model_ref: str
    hf_repo_id: str | None
    source_path: str | None
    source_backend: str | None
    source_is_quantized: bool
    source_sha256: str | None
    catalog_sha256: str
    n_layers: int
    n_tensors: int
    n_groups: int
    n_quantizable: int
    tensors: dict[str, CatalogTensor]
    groups: dict[str, CatalogGroup]
    steps_log: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_ref": self.model_ref,
            "hf_repo_id": self.hf_repo_id,
            "source_path": self.source_path,
            "source_backend": self.source_backend,
            "source_is_quantized": self.source_is_quantized,
            "source_sha256": self.source_sha256,
            "catalog_sha256": self.catalog_sha256,
            "n_layers": self.n_layers,
            "n_tensors": self.n_tensors,
            "n_groups": self.n_groups,
            "n_quantizable": self.n_quantizable,
            "tensors": {k: v.to_dict() for k, v in self.tensors.items()},
            "groups": {k: v.to_dict() for k, v in self.groups.items()},
            "steps_log": self.steps_log,
            "notes": self.notes,
        }

    def summary_dict(self) -> dict[str, Any]:
        return {
            "model_ref": self.model_ref,
            "hf_repo_id": self.hf_repo_id,
            "source_path": self.source_path,
            "source_backend": self.source_backend,
            "source_is_quantized": self.source_is_quantized,
            "source_sha256": self.source_sha256,
            "catalog_sha256": self.catalog_sha256,
            "n_layers": self.n_layers,
            "n_tensors": self.n_tensors,
            "n_groups": self.n_groups,
            "n_quantizable": self.n_quantizable,
            "group_ids": list(self.groups.keys()),
            "sample_tensors": {
                k: v.to_dict()
                for i, (k, v) in enumerate(self.tensors.items())
                if i < 8
            },
            "steps_log": self.steps_log,
            "notes": self.notes,
        }

# --- from catalog/names.py ---
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

# --- from catalog/catalog.py ---
def build_catalog(
    classified_tensors: list[dict[str, Any]],
    *,
    model_ref: str,
    hf_repo_id: str | None = None,
    source_path: str | None = None,
    source_backend: str | None = None,
    source_is_quantized: bool = False,
    source_sha256: str | None = None,
    n_layers: int | None = None,
) -> Catalog:
    log: list[str] = []
    log.append(f"1. Building catalog from {len(classified_tensors)} classified tensors")
    log.append(f"2. Source backend={source_backend!r} path={source_path}")

    tensors: dict[str, CatalogTensor] = {}
    group_acc: dict[str, list[CatalogTensor]] = defaultdict(list)

    for t in classified_tensors:
        raw_name = t["name"]
        if looks_like_gguf_name(raw_name) or (source_backend == "gguf"):
            gguf_name = raw_name
            hf_name = gguf_to_hf(raw_name)
            primary = gguf_name
        else:
            hf_name = raw_name
            gguf_name = hf_to_gguf(raw_name)
            primary = hf_name

        ct = CatalogTensor(
            name=primary,
            gguf_name=gguf_name,
            hf_name=hf_name,
            shape=list(t.get("shape") or []),
            dtype=str(t.get("dtype") or "unknown"),
            role=str(t["role"]),
            layer=t.get("layer"),
            depth=t.get("depth"),
            group_id=str(t["group_id"]),
            nbytes=int(t.get("nbytes") or 0),
            n_elements=int(t.get("n_elements") or 0),
            quantizable=bool(t.get("quantizable")),
            weight_features=None,
            activation_features=None,
        )
        tensors[primary] = ct
        group_acc[ct.group_id].append(ct)

    groups: dict[str, CatalogGroup] = {}
    for gid, members in sorted(group_acc.items()):
        role = members[0].role
        depth = members[0].depth
        groups[gid] = CatalogGroup(
            group_id=gid,
            role=role,
            depth=depth,
            quantizable=any(m.quantizable for m in members),
            n_tensors=len(members),
            total_nbytes=sum(m.nbytes for m in members),
            tensor_names=sorted(m.name for m in members),
        )

    if n_layers is None:
        layers = [t.layer for t in tensors.values() if t.layer is not None]
        n_layers = (max(layers) + 1) if layers else 0

    n_quant = sum(1 for t in tensors.values() if t.quantizable)
    log.append(f"3. Catalog tensors={len(tensors)} groups={len(groups)} quantizable={n_quant}")
    mapped_hf = sum(1 for t in tensors.values() if t.hf_name)
    mapped_gguf = sum(1 for t in tensors.values() if t.gguf_name)
    log.append(f"4. Name maps filled: hf_name={mapped_hf}/{len(tensors)} gguf_name={mapped_gguf}/{len(tensors)}")

    # Hash canonical payload (without hash field / logs)
    preimage = {
        "model_ref": model_ref,
        "hf_repo_id": hf_repo_id,
        "source_path": source_path,
        "source_backend": source_backend,
        "n_layers": n_layers,
        "tensors": {k: v.to_dict() for k, v in sorted(tensors.items())},
        "groups": {k: v.to_dict() for k, v in sorted(groups.items())},
    }
    digest = hashlib.sha256(
        json.dumps(preimage, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    log.append(f"5. catalog_sha256={digest[:16]}…")
    log.append("6. Catalog ready — weight features are Step 06 (slots are null for now)")

    notes = [
        "weight_features / activation_features are null until Steps 06 / 08.",
        "group_id is the probe unit for Step 12.",
    ]
    if source_backend == "gguf":
        notes.append(
            "Primary keys are GGUF names (Ollama source). hf_name is best-effort reverse map."
        )

    return Catalog(
        model_ref=model_ref,
        hf_repo_id=hf_repo_id,
        source_path=source_path,
        source_backend=source_backend,
        source_is_quantized=source_is_quantized,
        source_sha256=source_sha256,
        catalog_sha256=digest,
        n_layers=n_layers,
        n_tensors=len(tensors),
        n_groups=len(groups),
        n_quantizable=n_quant,
        tensors=tensors,
        groups=groups,
        steps_log=log,
        notes=notes,
    )
