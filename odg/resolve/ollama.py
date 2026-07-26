"""Read local Ollama manifests / GGUF metadata. Never treat Ollama blobs as BF16 sources."""

from __future__ import annotations

import json
import re
import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .maps import lookup_ollama_hf, ollama_name_from_tag


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
