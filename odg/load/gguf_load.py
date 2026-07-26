"""GGUF open + header parse for Step 02 (Ollama / llama.cpp sources)."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

# ggml_type → (name, block_size, type_size) for nbytes estimate
# See ggml-common.h — simplified for common quant types.
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 40),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
}


def _read_str(f) -> str:
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", errors="replace")


def _skip_val(f, t: int) -> None:
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if t == 8:
        _read_str(f)
    elif t == 9:
        at = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        if at == 8:
            for _ in range(n):
                _read_str(f)
        else:
            f.read(sizes.get(at, 0) * n)
    else:
        f.read(sizes[t])


def _read_val(f, t: int) -> Any:
    if t == 4:
        return struct.unpack("<I", f.read(4))[0]
    if t == 5:
        return struct.unpack("<i", f.read(4))[0]
    if t == 6:
        return struct.unpack("<f", f.read(4))[0]
    if t == 7:
        return bool(f.read(1)[0])
    if t == 8:
        return _read_str(f)
    if t == 10:
        return struct.unpack("<Q", f.read(8))[0]
    if t == 11:
        return struct.unpack("<q", f.read(8))[0]
    if t == 12:
        return struct.unpack("<d", f.read(8))[0]
    if t == 9:
        at = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        # Skip payload but remember length (vocab etc. can be huge)
        if at == 8:
            for _ in range(n):
                _read_str(f)
        else:
            f.read(sizes.get(at, 0) * n)
        return {"_array_len": int(n), "_array_type": int(at)}
    _skip_val(f, t)
    return None


def _approx_nbytes(n_elements: int, ggml_type: int) -> int | None:
    info = GGML_TYPES.get(ggml_type)
    if not info:
        return None
    _name, block, tsize = info
    if block <= 1:
        return n_elements * tsize
    n_blocks = (n_elements + block - 1) // block
    return n_blocks * tsize


def open_gguf(path: Path) -> dict[str, Any]:
    """
    Open a GGUF file and parse metadata + tensor index.

    Does **not** decode weight payloads into RAM — only the header/index.
    That is enough for Step 02 "load" with an Ollama source and for Step 03.
    """
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"GGUF not found: {path}")

    file_size = path.stat().st_size
    metadata: dict[str, Any] = {}
    tensors: list[dict[str, Any]] = []

    with path.open("rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Not a GGUF file (magic={magic!r}): {path}")
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        for _ in range(n_kv):
            key = _read_str(f)
            t = struct.unpack("<I", f.read(4))[0]
            metadata[key] = _read_val(f, t)

        for _ in range(n_tensors):
            name = _read_str(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            # GGUF stores dims in reverse order vs PyTorch convention often;
            # keep as stored, also provide shape reversed for readability.
            ggml_type = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            shape = list(reversed(dims))
            n_elem = 1
            for d in dims:
                n_elem *= d
            type_name, _, _ = GGML_TYPES.get(ggml_type, (f"TYPE_{ggml_type}", 1, 0))
            tensors.append(
                {
                    "name": name,
                    "shape": shape,
                    "dims_raw": dims,
                    "dtype": type_name,
                    "ggml_type": ggml_type,
                    "offset": offset,
                    "n_elements": n_elem,
                    "nbytes_approx": _approx_nbytes(n_elem, ggml_type),
                }
            )

    arch = metadata.get("general.architecture")
    param_count = metadata.get("general.parameter_count")
    layer_count = None
    embedding_length = None
    context_length = None
    vocab_size = None
    tok = metadata.get("tokenizer.ggml.tokens")
    if isinstance(tok, dict) and "_array_len" in tok:
        vocab_size = int(tok["_array_len"])
    elif isinstance(tok, list):
        vocab_size = len(tok)

    for k, v in metadata.items():
        if k.endswith(".block_count") and isinstance(v, int):
            layer_count = v
        if k.endswith(".embedding_length") and isinstance(v, int):
            embedding_length = v
        if k.endswith(".context_length") and isinstance(v, int):
            context_length = v

    dtype_summary: dict[str, int] = {}
    for t in tensors:
        dtype_summary[t["dtype"]] = dtype_summary.get(t["dtype"], 0) + 1

    # Keep metadata JSON-friendly (drop huge/non-serializable values)
    meta_out: dict[str, Any] = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            meta_out[k] = v
        elif isinstance(v, dict) and "_array_len" in v:
            meta_out[k] = v

    return {
        "path": str(path),
        "file_size_bytes": file_size,
        "gguf_version": version,
        "n_tensors": n_tensors,
        "architecture": str(arch) if arch is not None else None,
        "parameter_count": int(param_count) if isinstance(param_count, int) else None,
        "layer_count": layer_count,
        "embedding_length": embedding_length,
        "context_length": context_length,
        "vocab_size": vocab_size,
        "metadata": meta_out,
        "tensors": tensors,
        "dtype_summary": dtype_summary,
    }
