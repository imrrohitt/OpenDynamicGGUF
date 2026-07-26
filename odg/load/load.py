"""
Step 02 — Load the model resolved in Step 01.

For Ollama GGUF (current default): open the blob, parse header + tensor index
(mmap-friendly — does not decode all weights into RAM).

For HF directories: index safetensors / read config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .gguf_load import open_gguf
from .hf_load import open_hf_dir
from .types import LoadedModel, TensorInfo


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
