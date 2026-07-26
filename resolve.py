"""Step 01 — resolve any model reference to original BF16 source metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any
import re
from pathlib import Path
import json
import struct
import subprocess
from dataclasses import dataclass, field


# --- from resolve/types.py ---
class SourceKind(str, Enum):
    """How the user named the model."""

    HF = "hf"
    OLLAMA = "ollama"
    MLX = "mlx"
    LOCAL = "local"


@dataclass
class ArchitectureDescriptor:
    """What we know about the model after resolving (before full load)."""

    family: str | None = None
    layer_count: int | None = None
    embedding_length: int | None = None
    parameter_count: int | None = None
    context_length: int | None = None
    is_moe: bool = False
    is_hybrid_ssm: bool = False
    chat_template: str | None = None
    specialty_domain: str | None = None
    # Provenance / warnings
    ollama_quantization: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResolvedModel:
    """
    Result of Step 01.

    For Ollama (default right now):
      - ``local_path`` is the local Ollama GGUF blob
      - ``source_is_quantized`` is True if that blob is Q4/Q8/…
      - ``hf_repo_id`` is still recorded as the ideal BF16 upstream for later

    For HF / local BF16:
      - ``local_path`` points at full-precision weights when available
    """

    user_ref: str
    kind: SourceKind
    hf_repo_id: str | None
    local_path: str | None
    weights_ready: bool
    source_sha256: str | None
    descriptor: ArchitectureDescriptor
    source_is_quantized: bool = False
    rejected_quantized_source: str | None = None
    steps_log: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_ref": self.user_ref,
            "kind": self.kind.value,
            "hf_repo_id": self.hf_repo_id,
            "local_path": self.local_path,
            "weights_ready": self.weights_ready,
            "source_sha256": self.source_sha256,
            "source_is_quantized": self.source_is_quantized,
            "rejected_quantized_source": self.rejected_quantized_source,
            "descriptor": self.descriptor.to_dict(),
            "steps_log": self.steps_log,
        }

# --- from resolve/maps.py ---
OLLAMA_TO_HF: dict[str, str] = {
    "functiongemma": "google/functiongemma-270m-it",
    "gemma3": "google/gemma-3-270m-it",
    "gemma2": "google/gemma-2-2b-it",
    "gemma": "google/gemma-2-2b-it",
    "llama3.2": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.1": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-coder": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "phi4": "microsoft/phi-4",
}

# MLX-style ids / tags → HF BF16 source (never use the MLX quantized weights).
MLX_TO_HF: dict[str, str] = {
    "gemma4:e2b-mlx": "google/gemma-3-4b-it",  # placeholder family mapping; refine when gemma4 ships
    "mlx-community/gemma-2-2b-it-4bit": "google/gemma-2-2b-it",
}


def ollama_name_from_tag(tag: str) -> str:
    """functiongemma:latest → functiongemma"""
    return tag.split(":", 1)[0].strip().lower()


def lookup_ollama_hf(tag: str) -> str | None:
    name = ollama_name_from_tag(tag)
    if name in OLLAMA_TO_HF:
        return OLLAMA_TO_HF[name]
    # Heuristic: functiongemma → google/functiongemma-270m-it style already covered;
    # try google/<name> for gemma-family names.
    if "gemma" in name or name.startswith("function"):
        # Prefer instruct / it suffix when unknown size.
        return f"google/{name}-270m-it" if "270" not in name and "functiongemma" in name else f"google/{name}"
    return None


def lookup_mlx_hf(ref: str) -> str | None:
    key = ref.strip().lower()
    if key in MLX_TO_HF:
        return MLX_TO_HF[key]
    # Strip -mlx / :mlx suffixes and hope for a known ollama/hf name.
    bare = key.replace(":mlx", "").replace("-mlx", "").removesuffix(":latest")
    return lookup_ollama_hf(bare)

# --- from resolve/classify.py ---
_HF_REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_OLLAMA_TAG = re.compile(r"^[A-Za-z0-9._-]+:[A-Za-z0-9._-]+$")
_OLLAMA_BARE = re.compile(r"^[A-Za-z0-9._-]+$")


def classify_ref(user_ref: str) -> SourceKind:
    """
    Decide what kind of reference the user passed.

    Order matters:
      1. Existing local path → LOCAL
      2. Explicit MLX markers → MLX
      3. org/name → HF
      4. name:tag → OLLAMA
      5. bare name that exists in Ollama library heuristics → OLLAMA
      6. otherwise treat bare names as OLLAMA-style tags (common UX)
    """
    ref = user_ref.strip()
    path = Path(ref).expanduser()

    if path.exists() and path.is_dir():
        return SourceKind.LOCAL

    lower = ref.lower()
    if lower.endswith("-mlx") or lower.endswith(":mlx") or ":mlx" in lower or lower.startswith("mlx-community/"):
        return SourceKind.MLX

    if _HF_REPO.match(ref) and not ref.lower().endswith(":latest"):
        # mlx-community/foo already caught above
        return SourceKind.HF

    if _OLLAMA_TAG.match(ref):
        return SourceKind.OLLAMA

    # Bare names like "functiongemma" are almost always Ollama tags in this UX.
    if _OLLAMA_BARE.match(ref):
        return SourceKind.OLLAMA

    raise ValueError(
        f"Cannot classify model reference {user_ref!r}. "
        "Use an HF id (google/...), Ollama tag (functiongemma:latest), "
        "MLX id, or a local directory of safetensors."
    )

# --- from resolve/local.py ---
def inspect_local_dir(path: Path) -> tuple[ArchitectureDescriptor, bool]:
    """
    Validate a local checkpoint directory.

    Returns (descriptor, looks_full_precision).
    """
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Local path is not a directory: {path}")

    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"No config.json in {path}")

    config = json.loads(config_path.read_text())
    safes = list(path.glob("*.safetensors")) + list(path.glob("model*.safetensors"))
    bins = list(path.glob("pytorch_model*.bin"))
    if not safes and not bins:
        raise FileNotFoundError(
            f"No safetensors / pytorch_model*.bin weights in {path}"
        )

    # Heuristic: bitsandbytes / quantized configs
    quant_cfg = config.get("quantization_config")
    looks_quantized = quant_cfg is not None

    arch_list = config.get("architectures") or []
    arch0 = arch_list[0] if arch_list else str(config.get("model_type", "unknown"))
    family = str(config.get("model_type") or arch0).lower()

    desc = ArchitectureDescriptor(
        family=family,
        layer_count=config.get("num_hidden_layers"),
        embedding_length=config.get("hidden_size"),
        context_length=config.get("max_position_embeddings"),
        is_moe=bool(config.get("num_local_experts") or config.get("num_experts")),
        chat_template=_guess_chat_template(path, family),
        specialty_domain=_guess_specialty(path, family),
    )
    if looks_quantized:
        desc.notes.append(
            "config.json contains quantization_config — this directory may not be BF16."
        )
    return desc, not looks_quantized


def _guess_chat_template(path: Path, family: str) -> str | None:
    tok = path / "tokenizer_config.json"
    if tok.is_file():
        data = json.loads(tok.read_text())
        if data.get("chat_template"):
            return family
    return family or None


def _guess_specialty(path: Path, family: str) -> str | None:
    name = path.name.lower()
    if "function" in name or "tool" in name:
        return "function_calling"
    return None

# --- from resolve/hf.py ---
def fetch_hf_config(repo_id: str, token: str | None = None) -> dict[str, Any]:
    """Download only config.json from the Hub (lightweight)."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo_id,
        filename="config.json",
        token=token,
    )
    return json.loads(Path(path).read_text())


def descriptor_from_hf_config(
    config: dict[str, Any],
    repo_id: str,
    specialty_hint: str | None = None,
) -> ArchitectureDescriptor:
    arch_list = config.get("architectures") or []
    arch0 = arch_list[0] if arch_list else str(config.get("model_type", "unknown"))
    family = str(config.get("model_type") or arch0).lower()
    specialty = specialty_hint
    if specialty is None and "function" in repo_id.lower():
        specialty = "function_calling"

    return ArchitectureDescriptor(
        family=family,
        layer_count=config.get("num_hidden_layers"),
        embedding_length=config.get("hidden_size"),
        context_length=config.get("max_position_embeddings"),
        is_moe=bool(config.get("num_local_experts") or config.get("num_experts")),
        chat_template=family,
        specialty_domain=specialty,
    )


def try_prepare_hf_weights(
    repo_id: str,
    cache_dir: Path,
    *,
    download_weights: bool,
    token: str | None = None,
) -> tuple[str | None, bool, str | None, list[str]]:
    """
    Optionally download the HF snapshot.

    Returns (local_path, weights_ready, error_message, notes).
    """
    notes: list[str] = []
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return None, False, "huggingface_hub is not installed", notes

    try:
        if download_weights:
            local = snapshot_download(
                repo_id=repo_id,
                cache_dir=str(cache_dir),
                token=token,
                ignore_patterns=["*.gguf", "*.bin.gz"],
            )
            notes.append(f"Downloaded full snapshot of {repo_id} to {local}")
            return local, True, None, notes

        # Config-only: still useful; mark weights not ready.
        from huggingface_hub import hf_hub_download

        cfg = hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            cache_dir=str(cache_dir),
            token=token,
        )
        local = str(Path(cfg).parent)
        notes.append(
            f"Fetched config.json for {repo_id}. "
            "Full BF16 weights not downloaded yet "
            "(pass --download-weights after `huggingface-cli login`)."
        )
        return local, False, None, notes
    except Exception as exc:  # noqa: BLE001 — surface Hub/gated errors clearly
        msg = str(exc)
        if "gated" in msg.lower() or "401" in msg or "403" in msg:
            msg = (
                f"Hugging Face repo {repo_id} is gated or needs authentication.\n"
                "  1. Visit the model page and accept the license\n"
                "  2. Run: huggingface-cli login\n"
                "  3. Re-run: odg resolve --model … --download-weights\n"
                f"Original error: {exc}"
            )
        return None, False, msg, notes

# --- from resolve/ollama.py ---
@dataclass
class OllamaInspectResult:
    tag: str
    architecture: str | None = None
    parameter_count: int | None = None
    quantization: str | None = None
    context_length: int | None = None
    embedding_length: int | None = None
    layer_count: int | None = None
    capabilities: list[str] = field(default_factory=list)
    model_blob_path: Path | None = None
    config_blob: dict | None = None
    upstream_hf: str | None = None
    is_quantized: bool = True
    raw_show: str = ""


def default_ollama_root() -> Path:
    return Path.home() / ".ollama" / "models"


def _parse_show_text(text: str) -> dict[str, str]:
    """Parse `ollama show` human-readable output into a flat dict."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # "architecture        gemma3"
        m = re.match(r"^([A-Za-z][A-Za-z0-9_ ]+?)\s{2,}(.+)$", line)
        if m:
            key = re.sub(r"\s+", "_", m.group(1).strip().lower())
            out[key] = m.group(2).strip()
    return out


def run_ollama_show(tag: str) -> str:
    proc = subprocess.run(
        ["ollama", "show", tag],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise FileNotFoundError(
            f"Ollama could not show {tag!r}. Is the model pulled?\n{err}"
        )
    return proc.stdout


def find_manifest(tag: str, ollama_root: Path | None = None) -> Path:
    root = ollama_root or default_ollama_root()
    name = ollama_name_from_tag(tag)
    version = tag.split(":", 1)[1] if ":" in tag else "latest"
    candidates = [
        root / "manifests" / "registry.ollama.ai" / "library" / name / version,
        root / "manifests" / "registry.ollama.ai" / "library" / name / "latest",
    ]
    for c in candidates:
        if c.is_file():
            return c
    # Search under library/<name>/
    lib = root / "manifests" / "registry.ollama.ai" / "library" / name
    if lib.is_dir():
        files = sorted(lib.iterdir())
        if files:
            return files[0]
    raise FileNotFoundError(f"No local Ollama manifest for {tag!r} under {root}")


def _digest_to_blob(root: Path, digest: str) -> Path:
    # digest like "sha256:abc..." → blobs/sha256-abc...
    h = digest.replace(":", "-", 1)
    return root / "blobs" / h


def read_gguf_metadata(path: Path, max_keys: int = 64) -> dict[str, object]:
    """Parse GGUF key/value metadata (not tensor weights)."""
    interesting: dict[str, object] = {}

    def read_str(f) -> str:
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8", errors="replace")

    def skip_val(f, t: int) -> None:
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if t == 8:
            read_str(f)
        elif t == 9:
            at = struct.unpack("<I", f.read(4))[0]
            n = struct.unpack("<Q", f.read(8))[0]
            if at == 8:
                for _ in range(n):
                    read_str(f)
            else:
                f.read(sizes.get(at, 0) * n)
        else:
            f.read(sizes[t])

    def read_val(f, t: int):
        if t == 4:
            return struct.unpack("<I", f.read(4))[0]
        if t == 5:
            return struct.unpack("<i", f.read(4))[0]
        if t == 6:
            return struct.unpack("<f", f.read(4))[0]
        if t == 7:
            return bool(f.read(1)[0])
        if t == 8:
            return read_str(f)
        if t == 10:
            return struct.unpack("<Q", f.read(8))[0]
        if t == 11:
            return struct.unpack("<q", f.read(8))[0]
        if t == 12:
            return struct.unpack("<d", f.read(8))[0]
        skip_val(f, t)
        return None

    with path.open("rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            return {}
        _version = struct.unpack("<I", f.read(4))[0]
        _n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        for _ in range(min(n_kv, max_keys * 4)):
            key = read_str(f)
            t = struct.unpack("<I", f.read(4))[0]
            keep = key.startswith("general.") or ".block_count" in key or ".context_length" in key or ".embedding_length" in key or key.endswith(".architecture")
            if keep:
                interesting[key] = read_val(f, t)
            else:
                skip_val(f, t)
            if len(interesting) >= max_keys:
                break
    return interesting


def inspect_ollama(tag: str, ollama_root: Path | None = None) -> OllamaInspectResult:
    """
    Inspect a locally pulled Ollama model.

    Returns metadata for the descriptor, and marks the blob as quantized
    so the resolver refuses to use it as a quantization source.
    """
    root = ollama_root or default_ollama_root()
    show = run_ollama_show(tag)
    fields = _parse_show_text(show)

    result = OllamaInspectResult(
        tag=tag,
        architecture=fields.get("architecture"),
        quantization=fields.get("quantization"),
        context_length=_maybe_int(fields.get("context_length")),
        embedding_length=_maybe_int(fields.get("embedding_length")),
        raw_show=show,
        upstream_hf=lookup_ollama_hf(tag),
    )

    # Capabilities lines after header are free-form; detect tools.
    if re.search(r"^\s*tools\s*$", show, re.M | re.I) or "tools" in show.lower():
        result.capabilities.append("tools")

    params = fields.get("parameters")
    if params:
        result.parameter_count = _parse_param_count(params)

    manifest_path = find_manifest(tag, root)
    manifest = json.loads(manifest_path.read_text())
    config_digest = manifest.get("config", {}).get("digest")
    if config_digest:
        cfg_path = _digest_to_blob(root, config_digest)
        if cfg_path.is_file():
            result.config_blob = json.loads(cfg_path.read_text())
            ft = result.config_blob.get("file_type")
            if ft:
                result.quantization = result.quantization or str(ft)
            fam = result.config_blob.get("model_family")
            if fam:
                result.architecture = result.architecture or str(fam)

    for layer in manifest.get("layers", []):
        if layer.get("mediaType") == "application/vnd.ollama.image.model":
            blob = _digest_to_blob(root, layer["digest"])
            result.model_blob_path = blob
            if blob.is_file():
                meta = read_gguf_metadata(blob)
                result.architecture = result.architecture or _as_str(meta.get("general.architecture"))
                pc = meta.get("general.parameter_count")
                if isinstance(pc, int):
                    result.parameter_count = pc
                # family.block_count
                for k, v in meta.items():
                    if k.endswith(".block_count") and isinstance(v, int):
                        result.layer_count = v
                    if k.endswith(".context_length") and isinstance(v, int):
                        result.context_length = result.context_length or v
                    if k.endswith(".embedding_length") and isinstance(v, int):
                        result.embedding_length = result.embedding_length or v
            break

    # Anything that is not F16/BF16/F32 is a quantized source → reject as input.
    q = (result.quantization or "").upper()
    result.is_quantized = q not in {"", "F16", "FP16", "BF16", "F32", "FP32"}
    return result


def _maybe_int(v: str | None) -> int | None:
    if v is None:
        return None
    digits = re.sub(r"[^0-9]", "", v)
    return int(digits) if digits else None


def _parse_param_count(s: str) -> int | None:
    # "268.10M" → 268100000 approx
    m = re.match(r"([0-9.]+)\s*([KMB])", s.strip(), re.I)
    if not m:
        return _maybe_int(s)
    num = float(m.group(1))
    unit = m.group(2).upper()
    mult = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}[unit]
    return int(num * mult)


def _as_str(v: object) -> str | None:
    return str(v) if v is not None else None

# --- from resolve/resolve.py ---
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
            "Add it to resolve.OLLAMA_TO_HF or pass the HF id directly."
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
