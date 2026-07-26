"""HF / safetensors load path for Step 02 (when resolve used --prefer-hf)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
