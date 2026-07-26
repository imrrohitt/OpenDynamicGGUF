"""
Step 05 — Assemble tensor_catalog.json (source of truth for probes/export).
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any

from .names import gguf_to_hf, hf_to_gguf, looks_like_gguf_name
from .types import Catalog, CatalogGroup, CatalogTensor


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
