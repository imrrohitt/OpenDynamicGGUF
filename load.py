"""Step 02 — load the resolved model (GGUF or HF) for inspection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal
import struct
from pathlib import Path
from typing import Any
import json


# --- from load/types.py ---
LoadBackend = Literal["gguf", "hf_safetensors", "hf_transformers"]


@dataclass
class TensorInfo:
    """Lightweight tensor descriptor (full enumerate is Step 03)."""

    name: str
    shape: list[int]
    dtype: str
    nbytes_approx: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LoadedModel:
    """
    Result of Step 02.

    We do not keep a live PyTorch module in the checkpoint — that cannot be
    serialized. Instead we record everything later steps need to reopen the
    same source (path, backend, metadata, tensor index summary).
    """

    backend: LoadBackend
    source_path: str
    source_is_quantized: bool
    architecture: str | None
    n_tensors: int
    file_size_bytes: int
    parameter_count: int | None
    layer_count: int | None
    embedding_length: int | None
    context_length: int | None
    vocab_size: int | None
    dtype_summary: dict[str, int] = field(default_factory=dict)
    sample_tensors: list[TensorInfo] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    steps_log: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "source_path": self.source_path,
            "source_is_quantized": self.source_is_quantized,
            "architecture": self.architecture,
            "n_tensors": self.n_tensors,
            "file_size_bytes": self.file_size_bytes,
            "parameter_count": self.parameter_count,
            "layer_count": self.layer_count,
            "embedding_length": self.embedding_length,
            "context_length": self.context_length,
            "vocab_size": self.vocab_size,
            "dtype_summary": self.dtype_summary,
            "sample_tensors": [t.to_dict() for t in self.sample_tensors],
            "metadata": self.metadata,
            "steps_log": self.steps_log,
            "notes": self.notes,
        }

# --- from load/gguf_load.py ---
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

# --- from load/hf_load.py ---
def open_hf_dir(path: Path) -> dict[str, Any]:
    """
    Open a local HF checkpoint directory.

    Prefers listing safetensors keys without loading full weights into GPU.
    Falls back to config-only if safetensors is unavailable.
    """
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"HF directory not found: {path}")

    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"No config.json in {path}")
    config = json.loads(config_path.read_text())

    tensors: list[dict[str, Any]] = []
    dtype_summary: dict[str, int] = {}
    backend = "hf_safetensors"
    notes: list[str] = []

    safe_files = sorted(path.glob("*.safetensors"))
    if safe_files:
        try:
            from safetensors import safe_open

            for sf in safe_files:
                with safe_open(str(sf), framework="pt") as f:
                    for key in f.keys():
                        slice_ = f.get_slice(key)
                        shape = list(slice_.get_shape())
                        dtype = str(slice_.get_dtype())
                        n_elem = 1
                        for d in shape:
                            n_elem *= d
                        # rough nbytes from dtype string
                        bpe = 2 if "16" in dtype else 4 if "32" in dtype else 1
                        tensors.append(
                            {
                                "name": key,
                                "shape": shape,
                                "dtype": dtype,
                                "n_elements": n_elem,
                                "nbytes_approx": n_elem * bpe,
                            }
                        )
                        dtype_summary[dtype] = dtype_summary.get(dtype, 0) + 1
            notes.append(f"Indexed {len(tensors)} tensors via safetensors (mmap).")
        except ImportError:
            backend = "hf_safetensors"
            notes.append(
                "safetensors package not installed — config loaded only. "
                "pip install safetensors"
            )
    else:
        notes.append("No *.safetensors files found — config only.")

    arch_list = config.get("architectures") or []
    return {
        "path": str(path),
        "file_size_bytes": sum(p.stat().st_size for p in path.rglob("*") if p.is_file()),
        "backend": backend,
        "architecture": str(config.get("model_type") or (arch_list[0] if arch_list else None)),
        "parameter_count": None,
        "layer_count": config.get("num_hidden_layers"),
        "embedding_length": config.get("hidden_size"),
        "context_length": config.get("max_position_embeddings"),
        "vocab_size": config.get("vocab_size"),
        "metadata": {
            "model_type": config.get("model_type"),
            "architectures": arch_list,
            "torch_dtype": config.get("torch_dtype"),
        },
        "tensors": tensors,
        "dtype_summary": dtype_summary,
        "n_tensors": len(tensors),
        "notes": notes,
        "source_is_quantized": config.get("quantization_config") is not None,
    }

# --- from load/load.py ---
def load_model(resolve_output: dict[str, Any]) -> LoadedModel:
    """
    Load from Step 01 ``output.json`` payload.

    Required keys: ``local_path``, ``source_is_quantized`` (optional),
    ``descriptor`` (optional).
    """
    log: list[str] = []
    path_str = resolve_output.get("local_path")
    if not path_str:
        raise ValueError(
            "Step 01 output has no local_path. Re-run: odg resolve --model …"
        )
    path = Path(path_str)
    log.append(f"1. Source path from resolve: {path}")

    source_is_quantized = bool(resolve_output.get("source_is_quantized"))
    desc = resolve_output.get("descriptor") or {}

    if path.is_file():
        log.append("2. Detected file source → opening as GGUF")
        return _load_gguf(path, source_is_quantized, desc, log)

    if path.is_dir():
        log.append("2. Detected directory source → opening as HF checkpoint")
        return _load_hf(path, source_is_quantized, desc, log)

    raise FileNotFoundError(f"Resolved path does not exist: {path}")


def _load_gguf(
    path: Path,
    source_is_quantized: bool,
    desc: dict[str, Any],
    log: list[str],
) -> LoadedModel:
    info = open_gguf(path)
    log.append(
        f"3. GGUF OK — version={info['gguf_version']}, "
        f"n_tensors={info['n_tensors']}, size={info['file_size_bytes']} bytes"
    )
    log.append(
        f"4. Architecture={info['architecture']!r}, "
        f"layers={info['layer_count']}, embed={info['embedding_length']}, "
        f"vocab={info['vocab_size']}"
    )
    log.append(f"5. Dtype mix: {info['dtype_summary']}")

    notes = []
    if source_is_quantized or any(
        k.startswith("Q") or k.startswith("IQ") for k in info["dtype_summary"]
    ):
        source_is_quantized = True
        notes.append(
            "Weights are quantized in-file (e.g. Q8_0). "
            "Step 02 indexes tensors without dequantizing into BF16 RAM."
        )

    samples = [
        TensorInfo(
            name=t["name"],
            shape=list(t["shape"]),
            dtype=t["dtype"],
            nbytes_approx=t.get("nbytes_approx"),
        )
        for t in info["tensors"][:12]
    ]
    log.append(
        f"6. Sample tensors (first {len(samples)}): "
        + ", ".join(t.name for t in samples[:5])
        + ("…" if len(samples) > 5 else "")
    )
    log.append("7. Load complete — handle is path + tensor index (checkpointed)")

    # Persist full tensor index beside step output via extra artifact in CLI;
    # include count here.
    return LoadedModel(
        backend="gguf",
        source_path=str(path),
        source_is_quantized=source_is_quantized,
        architecture=info["architecture"] or desc.get("family"),
        n_tensors=info["n_tensors"],
        file_size_bytes=info["file_size_bytes"],
        parameter_count=info["parameter_count"] or desc.get("parameter_count"),
        layer_count=info["layer_count"] or desc.get("layer_count"),
        embedding_length=info["embedding_length"] or desc.get("embedding_length"),
        context_length=info["context_length"] or desc.get("context_length"),
        vocab_size=info["vocab_size"],
        dtype_summary=info["dtype_summary"],
        sample_tensors=samples,
        metadata={
            "gguf_version": info["gguf_version"],
            "gguf_keys": sorted(info["metadata"].keys())[:40],
            "selected": {
                k: info["metadata"][k]
                for k in (
                    "general.architecture",
                    "general.parameter_count",
                    "general.file_type",
                    "general.name",
                )
                if k in info["metadata"]
            },
        },
        steps_log=log,
        notes=notes,
    )


def _load_hf(
    path: Path,
    source_is_quantized: bool,
    desc: dict[str, Any],
    log: list[str],
) -> LoadedModel:
    info = open_hf_dir(path)
    log.append(f"3. HF dir OK — n_tensors={info['n_tensors']}, size={info['file_size_bytes']}")
    for n in info.get("notes") or []:
        log.append(f"   · {n}")

    samples = [
        TensorInfo(
            name=t["name"],
            shape=list(t["shape"]),
            dtype=t["dtype"],
            nbytes_approx=t.get("nbytes_approx"),
        )
        for t in info["tensors"][:12]
    ]
    log.append("4. Load complete — safetensors index ready for Step 03")

    return LoadedModel(
        backend=info["backend"],  # type: ignore[arg-type]
        source_path=str(path),
        source_is_quantized=source_is_quantized or bool(info.get("source_is_quantized")),
        architecture=info["architecture"] or desc.get("family"),
        n_tensors=info["n_tensors"],
        file_size_bytes=info["file_size_bytes"],
        parameter_count=info.get("parameter_count") or desc.get("parameter_count"),
        layer_count=info.get("layer_count") or desc.get("layer_count"),
        embedding_length=info.get("embedding_length") or desc.get("embedding_length"),
        context_length=info.get("context_length") or desc.get("context_length"),
        vocab_size=info.get("vocab_size"),
        dtype_summary=info.get("dtype_summary") or {},
        sample_tensors=samples,
        metadata=info.get("metadata") or {},
        steps_log=log,
        notes=list(info.get("notes") or []),
    )


def tensor_index_from_resolve(resolve_output: dict[str, Any]) -> list[dict[str, Any]]:
    """Full tensor list for checkpoint artifact (used by Step 02 + 03)."""
    path = Path(resolve_output["local_path"])
    if path.is_file():
        return open_gguf(path)["tensors"]
    if path.is_dir():
        return open_hf_dir(path)["tensors"]
    raise FileNotFoundError(path)
