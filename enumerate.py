"""Step 03 — enumerate every tensor from Step 02's index."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import re
from collections import Counter


# --- from enumerate/types.py ---
@dataclass
class TensorRow:
    index: int
    name: str
    shape: list[int]
    dtype: str
    n_elements: int
    nbytes: int
    layer: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EnumerationResult:
    n_tensors: int
    total_elements: int
    total_nbytes: int
    dtype_summary: dict[str, int]
    layer_summary: dict[str, int]
    prefix_summary: dict[str, int]
    tensors: list[TensorRow]
    steps_log: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_tensors": self.n_tensors,
            "total_elements": self.total_elements,
            "total_nbytes": self.total_nbytes,
            "dtype_summary": self.dtype_summary,
            "layer_summary": self.layer_summary,
            "prefix_summary": self.prefix_summary,
            "tensors": [t.to_dict() for t in self.tensors],
            "steps_log": self.steps_log,
            "notes": self.notes,
        }

    def summary_dict(self) -> dict[str, Any]:
        """Compact output for output.json (full list lives in tensors.json)."""
        return {
            "n_tensors": self.n_tensors,
            "total_elements": self.total_elements,
            "total_nbytes": self.total_nbytes,
            "dtype_summary": self.dtype_summary,
            "layer_summary": self.layer_summary,
            "prefix_summary": self.prefix_summary,
            "sample_tensors": [t.to_dict() for t in self.tensors[:15]],
            "steps_log": self.steps_log,
            "notes": self.notes,
        }

# --- from enumerate/enumerate.py ---
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
