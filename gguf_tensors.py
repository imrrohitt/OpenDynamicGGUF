"""Read individual GGUF tensor payloads (F32 / F16 / BF16 / Q8_0) as float32."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

from load import GGML_TYPES, _approx_nbytes, _read_str, _read_val

GGUF_ALIGNMENT = 32


def _align_up(n: int, alignment: int = GGUF_ALIGNMENT) -> int:
    return (n + alignment - 1) // alignment * alignment


def gguf_tensor_map(path: Path) -> dict[str, Any]:
    """
    Parse GGUF header and return:
      data_offset — absolute file offset of tensor data section
      tensors     — name → {shape, ggml_type, dtype, offset, nbytes, n_elements}
    """
    path = path.expanduser().resolve()
    with path.open("rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Not a GGUF file: {path}")
        _version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        for _ in range(n_kv):
            _read_str(f)
            t = struct.unpack("<I", f.read(4))[0]
            _read_val(f, t)

        tensors: dict[str, dict[str, Any]] = {}
        for _ in range(n_tensors):
            name = _read_str(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            ggml_type = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            shape = list(reversed(dims))
            n_elem = 1
            for d in dims:
                n_elem *= d
            type_name, _, _ = GGML_TYPES.get(ggml_type, (f"TYPE_{ggml_type}", 1, 0))
            nbytes = _approx_nbytes(n_elem, ggml_type)
            tensors[name] = {
                "name": name,
                "shape": shape,
                "dims_raw": dims,
                "dtype": type_name,
                "ggml_type": ggml_type,
                "offset": offset,
                "n_elements": n_elem,
                "nbytes": nbytes,
            }

        data_offset = _align_up(f.tell())

    return {"path": str(path), "data_offset": data_offset, "tensors": tensors}


def _fp16_to_f32(u16: np.ndarray) -> np.ndarray:
    return u16.view(np.float16).astype(np.float32)


def _bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    # BF16: top 16 bits of float32
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def _dequant_q8_0(raw: bytes, n_elements: int) -> np.ndarray:
    """Q8_0: blocks of 32 int8 + fp16 scale (34 bytes)."""
    block = 34
    n_blocks = (n_elements + 31) // 32
    expected = n_blocks * block
    if len(raw) < expected:
        raise ValueError(f"Q8_0 short read: got {len(raw)} need {expected}")
    blocks = np.frombuffer(raw, dtype=np.uint8, count=expected).reshape(n_blocks, block)
    scales = blocks[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(-1)
    qs = blocks[:, 2:].copy().view(np.int8).astype(np.float32)
    return (qs * scales[:, None]).reshape(-1)[:n_elements].astype(np.float32, copy=False)


def read_tensor_f32(path: Path, info: dict[str, Any], data_offset: int) -> np.ndarray:
    """Load one tensor as a flat float32 array (dequantized if needed)."""
    path = path.expanduser().resolve()
    ggml_type = int(info["ggml_type"])
    n_elements = int(info["n_elements"])
    nbytes = info.get("nbytes") or _approx_nbytes(n_elements, ggml_type)
    if nbytes is None:
        raise ValueError(f"Unknown ggml_type {ggml_type} for {info.get('name')}")

    abs_off = data_offset + int(info["offset"])
    with path.open("rb") as f:
        f.seek(abs_off)
        raw = f.read(nbytes)

    if len(raw) != nbytes:
        raise ValueError(
            f"Short read for {info.get('name')}: got {len(raw)} expected {nbytes}"
        )

    if ggml_type == 0:  # F32
        arr = np.frombuffer(raw, dtype=np.float32, count=n_elements)
        return np.array(arr, dtype=np.float32, copy=True)
    if ggml_type == 1:  # F16
        u16 = np.frombuffer(raw, dtype=np.uint16, count=n_elements)
        return _fp16_to_f32(u16)
    if ggml_type == 30:  # BF16
        u16 = np.frombuffer(raw, dtype=np.uint16, count=n_elements)
        return _bf16_to_f32(u16)
    if ggml_type == 8:  # Q8_0
        return _dequant_q8_0(raw, n_elements)

    raise NotImplementedError(
        f"Dequant not implemented for dtype={info.get('dtype')} "
        f"(ggml_type={ggml_type}) — need BF16/F16/F32/Q8_0 for Step 06"
    )
