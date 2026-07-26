"""
Step 01 — Resolve any user model reference.

Default for Ollama tags (current workflow):
  Use the local Ollama GGUF as the working source so you can proceed
  without Hugging Face login.

Ideal production path (later / --prefer-hf):
  Map Ollama → original BF16 Hugging Face repo and download that.
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
    prefer_hf: bool = False,
    hf_token: str | None = None,
    ollama_root: str | Path | None = None,
) -> ResolvedModel:
    """
    Resolve ``user_ref``.

    Parameters
    ----------
    prefer_hf:
        If True (and kind is Ollama), try Hugging Face BF16 instead of the
        local Ollama blob. Default False = use Ollama locally right now.
    download_weights:
        Only used with prefer_hf / HF refs — download safetensors snapshot.
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
            user_ref,
            log,
            cache,
            download_weights=download_weights,
            prefer_hf=prefer_hf,
            hf_token=hf_token,
            ollama_root=ollama_root,
        )

    return _resolve_hf(user_ref, log, cache, download_weights, hf_token)


def _resolve_local(user_ref: str, log: list[str]) -> ResolvedModel:
    path = Path(user_ref).expanduser().resolve()
    log.append(f"2. Inspecting local directory {path}")
    desc, full_prec = inspect_local_dir(path)
    if not full_prec:
        raise ValueError(
            f"Local path {path} looks quantized (quantization_config present). "
            "Pass a full-precision checkpoint, or use an Ollama tag."
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
        source_is_quantized=False,
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
    *,
    download_weights: bool,
    prefer_hf: bool,
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

    if not info.model_blob_path or not info.model_blob_path.is_file():
        raise FileNotFoundError(
            f"Ollama model blob for {tag!r} not found on disk. "
            f"Run: ollama pull {tag}"
        )

    hf_id = info.upstream_hf or lookup_ollama_hf(tag)

    specialty = None
    step_n = 4
    if "tools" in info.capabilities or "function" in tag.lower():
        specialty = "function_calling"
        log.append(f"{step_n}. Detected tool/function-calling specialty domain")
        step_n += 1

    base = ArchitectureDescriptor(
        family=(info.architecture or "").lower() or None,
        layer_count=info.layer_count,
        embedding_length=info.embedding_length,
        parameter_count=info.parameter_count,
        context_length=info.context_length,
        chat_template=(info.architecture or "").lower() or None,
        specialty_domain=specialty,
        ollama_quantization=info.quantization,
        notes=[],
    )
    if info.layer_count:
        log.append(
            f"{step_n}. Enriched descriptor from GGUF metadata: "
            f"layers={info.layer_count}, embed={info.embedding_length}"
        )
        step_n += 1

    # --- Default path: use local Ollama GGUF ---
    if not prefer_hf:
        note = (
            f"Using local Ollama GGUF as working source "
            f"(quant={info.quantization}). "
            "Ideal later path is BF16 from Hugging Face "
            f"({hf_id or 'unknown'}); pass --prefer-hf when ready."
        )
        base.notes.append(note)
        log.append(f"{step_n}. USING Ollama blob as local_path: {info.model_blob_path}")
        step_n += 1
        log.append(f"{step_n}. {note}")
        step_n += 1
        if hf_id:
            log.append(f"{step_n}. Recorded upstream HF (for later): {hf_id}")

        return ResolvedModel(
            user_ref=user_ref,
            kind=SourceKind.OLLAMA,
            hf_repo_id=hf_id,
            local_path=str(info.model_blob_path),
            weights_ready=True,
            source_sha256=None,
            descriptor=base,
            source_is_quantized=bool(info.is_quantized),
            rejected_quantized_source=None,
            steps_log=log,
        )

    # --- Optional: prefer Hugging Face BF16 ---
    if not hf_id:
        raise ValueError(
            f"Could not map Ollama tag {tag!r} to an upstream HF repo. "
            "Add it to odg.resolve.maps.OLLAMA_TO_HF or pass the HF id directly."
        )
    rejected = None
    if info.is_quantized:
        rejected = (
            f"Ollama blob for {tag} is quantized ({info.quantization}). "
            "--prefer-hf is set, so we use Hugging Face BF16 instead of the blob."
        )
        log.append(f"{step_n}. Prefer HF: skipping quantized Ollama blob — {rejected}")
        step_n += 1
    log.append(f"{step_n}. Upstream full-precision HF repo: {hf_id}")
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
        source_is_quantized=False,
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
