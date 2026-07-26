"""
Step 03 — Enumerate every tensor.

Takes Step 02 ``tensor_index.json`` (or load output) and produces a clean,
sorted inventory with nbytes, layer ids, and summary tables.

Does not classify roles yet — that is Step 04.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .types import EnumerationResult, TensorRow

_LAYER_RE = re.compile(r"(?:blk|layers?)[.\[](\d+)", re.I)


def _layer_of(name: str) -> int | None:
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def _prefix_of(name: str) -> str:
    """blk.0.attn_q.weight → blk.*.attn_q.weight  (layer wildcard for grouping)."""
    return re.sub(r"(blk|layers?)[.\[]\d+[.\]]?", r"\1.*", name, count=1, flags=re.I)


def enumerate_tensors(tensor_index: list[dict[str, Any]]) -> EnumerationResult:
    log: list[str] = []
    log.append(f"1. Received tensor index with {len(tensor_index)} entries from Step 02")

    rows: list[TensorRow] = []
    dtype_c: Counter[str] = Counter()
    layer_c: Counter[str] = Counter()
    prefix_c: Counter[str] = Counter()
    total_elem = 0
    total_nbytes = 0

    for i, t in enumerate(tensor_index):
        name = t["name"]
        shape = list(t.get("shape") or [])
        dtype = str(t.get("dtype") or "unknown")
        n_elem = int(t.get("n_elements") or _prod(shape))
        nbytes = int(t.get("nbytes_approx") or t.get("nbytes") or 0)
        layer = _layer_of(name)

        rows.append(
            TensorRow(
                index=i,
                name=name,
                shape=shape,
                dtype=dtype,
                n_elements=n_elem,
                nbytes=nbytes,
                layer=layer,
            )
        )
        dtype_c[dtype] += 1
        layer_key = str(layer) if layer is not None else "global"
        layer_c[layer_key] += 1
        prefix_c[_prefix_of(name)] += 1
        total_elem += n_elem
        total_nbytes += nbytes

    # Stable sort by name for human-readable inventory (keep original index field)
    rows_sorted = sorted(rows, key=lambda r: r.name)

    log.append(f"2. Normalized {len(rows_sorted)} tensors (name, shape, dtype, nbytes)")
    log.append(f"3. Total elements={total_elem:,}  approx_bytes={total_nbytes:,}")
    log.append(f"4. Dtype breakdown: {dict(dtype_c)}")
    n_layers = len([k for k in layer_c if k != "global"])
    log.append(f"5. Layers touched: {n_layers}  (+ {layer_c.get('global', 0)} global tensors)")
    log.append("6. Enumeration complete — roles not assigned yet (Step 04)")

    notes = [
        "nbytes are approximate for quantized GGUF types (from ggml block sizes).",
        "Classification / quantizable flags come in Step 04.",
    ]

    return EnumerationResult(
        n_tensors=len(rows_sorted),
        total_elements=total_elem,
        total_nbytes=total_nbytes,
        dtype_summary=dict(dtype_c.most_common()),
        layer_summary=dict(sorted(layer_c.items(), key=lambda kv: (kv[0] == "global", int(kv[0]) if kv[0].isdigit() else 0))),
        prefix_summary=dict(prefix_c.most_common()),
        tensors=rows_sorted,
        steps_log=log,
        notes=notes,
    )


def _prod(shape: list[int]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def to_tsv(rows: list[TensorRow]) -> str:
    lines = ["index\tname\tshape\tdtype\tn_elements\tnbytes\tlayer"]
    for r in rows:
        shape = "x".join(str(d) for d in r.shape)
        layer = "" if r.layer is None else str(r.layer)
        lines.append(
            f"{r.index}\t{r.name}\t{shape}\t{r.dtype}\t{r.n_elements}\t{r.nbytes}\t{layer}"
        )
    return "\n".join(lines) + "\n"
