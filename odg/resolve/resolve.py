"""
Step 01 — Resolve any user model reference to the original BF16 source.

What this module does (and does NOT do):
  ✓ Classifies the reference (HF / Ollama / MLX / local)
  ✓ For Ollama/MLX: refuses quantized blobs as quantization sources
  ✓ Maps to the upstream Hugging Face full-precision repo
  ✓ Builds an ArchitectureDescriptor (family, layers, specialty, …)
  ✗ Does NOT load weights into a neural net (that's Step 02)
  ✗ Does NOT quantize anything
"""

from __future__ import annotations

from pathlib import Path

from .classify import classify_ref
from .hf import descriptor_from_hf_config, fetch_hf_config, try_prepare_hf_weights
from .local import inspect_local_dir
from .maps import lookup_mlx_hf, lookup_ollama_hf
from .ollama import inspect_ollama
from .types import ArchitectureDescriptor, ResolvedModel, SourceKind


def resolve_model(
    user_ref: str,
    *,
    cache_dir: str | Path | None = None,
    download_weights: bool = False,
    hf_token: str | None = None,
    ollama_root: str | Path | None = None,
) -> ResolvedModel:
    """
    Resolve ``user_ref`` to a full-precision source identity + descriptor.

    Parameters
    ----------
    user_ref:
        e.g. ``functiongemma:latest``, ``google/gemma-3-270m-it``, ``./my-model``
    download_weights:
        If True, attempt to download the HF safetensors snapshot.
        Default False keeps Step 01 fast and works even when only mapping
        Ollama → HF id (gated models still need login for download).
    """
    log: list[str] = []
    cache = Path(cache_dir or (Path.home() / ".cache" / "odg" / "models"))
    cache.mkdir(parents=True, exist_ok=True)

    kind = classify_ref(user_ref)
    log.append(f"1. Classified {user_ref!r} as {kind.value.upper()}")

    if kind == SourceKind.LOCAL:
        return _resolve_local(user_ref, log)

    if kind == SourceKind.MLX:
        return _resolve_mlx(user_ref, log, cache, download_weights, hf_token)

    if kind == SourceKind.OLLAMA:
        return _resolve_ollama(
            user_ref, log, cache, download_weights, hf_token, ollama_root
        )

    # HF
    return _resolve_hf(user_ref, log, cache, download_weights, hf_token)


def _resolve_local(user_ref: str, log: list[str]) -> ResolvedModel:
    path = Path(user_ref).expanduser().resolve()
    log.append(f"2. Inspecting local directory {path}")
    desc, full_prec = inspect_local_dir(path)
    if not full_prec:
        raise ValueError(
            f"Local path {path} looks quantized (quantization_config present). "
            "Step 01 requires original BF16/F16 weights."
        )
    log.append("3. Local checkpoint looks full-precision — using as source")
    return ResolvedModel(
        user_ref=user_ref,
        kind=SourceKind.LOCAL,
        hf_repo_id=None,
        local_path=str(path),
        weights_ready=True,
        source_sha256=None,
        descriptor=desc,
        steps_log=log,
    )


def _resolve_mlx(
    user_ref: str,
    log: list[str],
    cache: Path,
    download_weights: bool,
    hf_token: str | None,
) -> ResolvedModel:
    hf_id = lookup_mlx_hf(user_ref)
    rejected = (
        f"{user_ref} looks like an MLX (already-quantized) artifact. "
        "OpenDynamicGGUF will NOT requantize it."
    )
    log.append(f"2. REJECTED as quantization source: {rejected}")
    if not hf_id:
        raise ValueError(
            f"{rejected} Could not map to an upstream HF BF16 repo. "
            "Pass the HF id explicitly (org/model)."
        )
    log.append(f"3. Mapped MLX ref → upstream HF BF16 source: {hf_id}")
    return _finish_hf(
        user_ref=user_ref,
        kind=SourceKind.MLX,
        hf_id=hf_id,
        log=log,
        cache=cache,
        download_weights=download_weights,
        hf_token=hf_token,
        rejected=rejected,
        specialty_hint=None,
        base_desc=ArchitectureDescriptor(notes=[rejected]),
    )


def _resolve_ollama(
    user_ref: str,
    log: list[str],
    cache: Path,
    download_weights: bool,
    hf_token: str | None,
    ollama_root: str | Path | None,
) -> ResolvedModel:
    tag = user_ref if ":" in user_ref else f"{user_ref}:latest"
    log.append(f"2. Inspecting local Ollama model {tag}")
    info = inspect_ollama(tag, Path(ollama_root) if ollama_root else None)

    log.append(
        f"3. Ollama reports architecture={info.architecture!r}, "
        f"quantization={info.quantization!r}, "
        f"parameters={info.parameter_count}"
    )

    rejected = None
    if info.is_quantized:
        rejected = (
            f"Ollama blob for {tag} is quantized ({info.quantization}). "
            "Using it as a quantization source would stack quantization error. "
            "We resolve the original BF16 Hugging Face checkpoint instead."
        )
        log.append(f"4. REJECTED quantized Ollama blob as source — {rejected}")
        if info.model_blob_path:
            log.append(f"   (ignored blob: {info.model_blob_path})")
    else:
        log.append("4. Ollama blob appears full-precision (unusual); still prefer HF source")

    hf_id = info.upstream_hf or lookup_ollama_hf(tag)
    if not hf_id:
        raise ValueError(
            f"Could not map Ollama tag {tag!r} to an upstream HF repo. "
            "Add it to odg.resolve.maps.OLLAMA_TO_HF or pass the HF id directly."
        )
    log.append(f"5. Upstream full-precision HF repo: {hf_id}")

    specialty = None
    if "tools" in info.capabilities or "function" in tag.lower():
        specialty = "function_calling"
        log.append("6. Detected tool/function-calling specialty domain")

    base = ArchitectureDescriptor(
        family=(info.architecture or "").lower() or None,
        layer_count=info.layer_count,
        embedding_length=info.embedding_length,
        parameter_count=info.parameter_count,
        context_length=info.context_length,
        chat_template=(info.architecture or "").lower() or None,
        specialty_domain=specialty,
        ollama_quantization=info.quantization,
        notes=[rejected] if rejected else [],
    )
    if info.layer_count:
        log.append(
            f"7. Enriched descriptor from GGUF metadata: "
            f"layers={info.layer_count}, embed={info.embedding_length}"
        )

    return _finish_hf(
        user_ref=user_ref,
        kind=SourceKind.OLLAMA,
        hf_id=hf_id,
        log=log,
        cache=cache,
        download_weights=download_weights,
        hf_token=hf_token,
        rejected=rejected,
        specialty_hint=specialty,
        base_desc=base,
    )


def _resolve_hf(
    user_ref: str,
    log: list[str],
    cache: Path,
    download_weights: bool,
    hf_token: str | None,
) -> ResolvedModel:
    log.append(f"2. Treating {user_ref!r} as Hugging Face repo id")
    return _finish_hf(
        user_ref=user_ref,
        kind=SourceKind.HF,
        hf_id=user_ref,
        log=log,
        cache=cache,
        download_weights=download_weights,
        hf_token=hf_token,
        rejected=None,
        specialty_hint="function_calling" if "function" in user_ref.lower() else None,
        base_desc=ArchitectureDescriptor(),
    )


def _finish_hf(
    *,
    user_ref: str,
    kind: SourceKind,
    hf_id: str,
    log: list[str],
    cache: Path,
    download_weights: bool,
    hf_token: str | None,
    rejected: str | None,
    specialty_hint: str | None,
    base_desc: ArchitectureDescriptor,
) -> ResolvedModel:
    # Try to enrich from HF config when possible.
    try:
        cfg = fetch_hf_config(hf_id, token=hf_token)
        hf_desc = descriptor_from_hf_config(cfg, hf_id, specialty_hint)
        log.append(f"Fetched HF config.json for {hf_id}")
        desc = _merge_desc(base_desc, hf_desc)
    except Exception as exc:  # noqa: BLE001
        log.append(f"HF config not fetched yet ({exc.__class__.__name__}: {exc})")
        desc = base_desc
        if specialty_hint and not desc.specialty_domain:
            desc.specialty_domain = specialty_hint
        if not desc.family and "/" in hf_id:
            # weak fallback from repo name
            desc.family = hf_id.split("/", 1)[1].split("-")[0].lower()

    local_path, weights_ready, err, notes = try_prepare_hf_weights(
        hf_id,
        cache,
        download_weights=download_weights,
        token=hf_token,
    )
    desc.notes.extend(notes)
    if err:
        log.append(f"Weight prepare: {err}")
        desc.notes.append(err)
    else:
        log.append(
            "Weights ready"
            if weights_ready
            else "Identity resolved; BF16 weights download deferred"
        )

    return ResolvedModel(
        user_ref=user_ref,
        kind=kind,
        hf_repo_id=hf_id,
        local_path=local_path,
        weights_ready=weights_ready,
        source_sha256=None,
        descriptor=desc,
        rejected_quantized_source=rejected,
        steps_log=log,
    )


def _merge_desc(
    base: ArchitectureDescriptor, hf: ArchitectureDescriptor
) -> ArchitectureDescriptor:
    return ArchitectureDescriptor(
        family=hf.family or base.family,
        layer_count=hf.layer_count or base.layer_count,
        embedding_length=hf.embedding_length or base.embedding_length,
        parameter_count=base.parameter_count or hf.parameter_count,
        context_length=hf.context_length or base.context_length,
        is_moe=hf.is_moe or base.is_moe,
        is_hybrid_ssm=hf.is_hybrid_ssm or base.is_hybrid_ssm,
        chat_template=hf.chat_template or base.chat_template,
        specialty_domain=hf.specialty_domain or base.specialty_domain,
        ollama_quantization=base.ollama_quantization,
        notes=list(base.notes) + list(hf.notes),
    )
