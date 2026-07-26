"""
Step 09 — Freeze a GGUF reference for llama.cpp (imatrix / probes / export).

Ideal: HF BF16 → convert_hf_to_gguf.py --outtype bf16
Pragmatic: promote the already-resolved GGUF (e.g. Ollama Q8) with SHA + catalog check.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal

from odg.gguf_tensors import gguf_tensor_map
from odg.load.gguf_load import open_gguf

from .types import FreezeResult

Mode = Literal["auto", "hf-convert", "promote"]


def _sha256_file(path: Path, *, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def find_convert_script(explicit: str | Path | None = None) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.is_file() else None
    env = os.environ.get("LLAMA_CPP_DIR") or os.environ.get("LLAMA_CPP")
    candidates: list[Path] = []
    if env:
        root = Path(env).expanduser()
        candidates += [
            root / "convert_hf_to_gguf.py",
            root / "convert-hf-to-gguf.py",
        ]
    home = Path.home()
    candidates += [
        home / "llama.cpp" / "convert_hf_to_gguf.py",
        home / "src" / "llama.cpp" / "convert_hf_to_gguf.py",
        Path("/opt/homebrew/share/llama.cpp/convert_hf_to_gguf.py"),
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _is_hf_model_dir(path: Path | None) -> bool:
    if not path or not path.is_dir():
        return False
    # safetensors or bin + config
    has_cfg = (path / "config.json").is_file()
    weights = list(path.glob("*.safetensors")) + list(path.glob("pytorch_model*.bin"))
    return has_cfg and bool(weights)


def _link_or_copy(src: Path, dest: Path) -> str:
    """Prefer hardlink, then symlink, then copy. Returns method string."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        os.link(src, dest)
        return "hardlink"
    except OSError:
        pass
    try:
        os.symlink(src, dest)
        return "symlink"
    except OSError:
        pass
    shutil.copy2(src, dest)
    return "copy"


def _dtype_is_bf16_family(dtype_summary: dict[str, int]) -> bool:
    if not dtype_summary:
        return False
    # Pure BF16/F16 reference (F32 norms OK)
    allowed = {"BF16", "F16", "F32"}
    return all(k in allowed for k in dtype_summary) and (
        dtype_summary.get("BF16", 0) + dtype_summary.get("F16", 0) > 0
    )


def verify_catalog_tensors(
    gguf_path: Path, catalog_tensor_names: list[str]
) -> tuple[bool, list[str]]:
    gmap = gguf_tensor_map(gguf_path)
    have = set(gmap["tensors"])
    missing = [n for n in catalog_tensor_names if n not in have]
    return (len(missing) == 0, missing)


def convert_hf_to_bf16_gguf(
    *,
    hf_dir: Path,
    outfile: Path,
    convert_script: Path,
    outtype: str = "bf16",
) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python3",
        str(convert_script),
        str(hf_dir),
        "--outtype",
        outtype,
        "--outfile",
        str(outfile),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "convert_hf_to_gguf failed:\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout[-2000:]}\n"
            f"stderr:\n{proc.stderr[-2000:]}"
        )


def freeze_gguf(
    *,
    model_ref: str,
    out_dir: Path,
    source_path: str | Path | None,
    source_is_quantized: bool = False,
    hf_local_path: str | Path | None = None,
    catalog_tensor_names: list[str] | None = None,
    mode: Mode = "auto",
    convert_script: str | Path | None = None,
    require_bf16: bool = False,
) -> FreezeResult:
    """
    Produce a frozen GGUF under out_dir and return FreezeResult.

    Writes:
      model-bf16.gguf or model-ref.gguf
      *.sha256
      freeze_manifest.json (caller may also write)
    """
    log: list[str] = []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    hf_dir = Path(hf_local_path).expanduser() if hf_local_path else None
    src = Path(source_path).expanduser() if source_path else None
    script = find_convert_script(convert_script)

    method: str | None = None
    gguf_path: Path | None = None

    can_convert = (
        mode in {"auto", "hf-convert"}
        and _is_hf_model_dir(hf_dir)
        and script is not None
    )

    if mode == "hf-convert" and not can_convert:
        reasons = []
        if not _is_hf_model_dir(hf_dir):
            reasons.append("HF model directory with config+weights not available")
        if script is None:
            reasons.append(
                "convert_hf_to_gguf.py not found "
                "(set LLAMA_CPP_DIR or pass --convert-script)"
            )
        raise RuntimeError("hf-convert unavailable: " + "; ".join(reasons))

    if can_convert:
        assert hf_dir is not None and script is not None
        gguf_path = out_dir / "model-bf16.gguf"
        log.append(f"1. Converting HF → BF16 GGUF via {script}")
        log.append(f"2. HF dir: {hf_dir}")
        convert_hf_to_bf16_gguf(hf_dir=hf_dir, outfile=gguf_path, convert_script=script)
        method = "hf_convert_bf16"
        log.append(f"3. Wrote {gguf_path}")
    else:
        if mode == "auto":
            log.append(
                "1. No HF+convert_hf_to_gguf available — promoting source GGUF"
            )
        else:
            log.append("1. mode=promote — promoting source GGUF")

        if src is None or not src.is_file():
            raise FileNotFoundError(
                f"No GGUF source to promote (source_path={source_path!r})"
            )
        # Peek dtypes to name the file
        peek = open_gguf(src)
        is_bf16 = _dtype_is_bf16_family(peek.get("dtype_summary") or {})
        out_name = "model-bf16.gguf" if is_bf16 else "model-ref.gguf"
        gguf_path = out_dir / out_name
        link_method = _link_or_copy(src, gguf_path)
        method = "promote_source_gguf"
        log.append(f"2. Source: {src}")
        log.append(f"3. Linked/copied as {out_name} via {link_method}")
        if not is_bf16:
            notes.append(
                "Reference is NOT BF16 (promoted working GGUF, often Q8_0). "
                "For production probes: resolve --prefer-hf, install llama.cpp, "
                "re-run with --mode hf-convert."
            )

    assert gguf_path is not None and method is not None

    info = open_gguf(gguf_path)
    dtype_summary = dict(info.get("dtype_summary") or {})
    is_bf16_ref = method == "hf_convert_bf16" or _dtype_is_bf16_family(dtype_summary)
    sha = _sha256_file(gguf_path)
    sha_path = Path(str(gguf_path) + ".sha256")
    sha_path.write_text(f"{sha}  {gguf_path.name}\n", encoding="utf-8")
    log.append(f"4. sha256={sha[:24]}… size={info['file_size_bytes']} bytes")
    log.append(f"5. dtype_summary={dtype_summary}")

    missing: list[str] = []
    catalog_match = True
    if catalog_tensor_names:
        catalog_match, missing = verify_catalog_tensors(gguf_path, catalog_tensor_names)
        if catalog_match:
            log.append(
                f"6. Catalog tensors match GGUF ({len(catalog_tensor_names)} names)"
            )
        else:
            log.append(
                f"6. Catalog mismatch: {len(missing)} missing "
                f"(sample: {missing[:5]})"
            )
            notes.append(f"{len(missing)} catalog tensor name(s) missing from GGUF")

    if require_bf16 and not is_bf16_ref:
        raise RuntimeError(
            "require_bf16=True but frozen GGUF is not a BF16/F16 reference. "
            f"dtypes={dtype_summary}"
        )

    if is_bf16_ref:
        notes.append("BF16/F16 reference ready for llama.cpp imatrix / probes.")
    else:
        notes.append(
            "Downstream steps can use this file for plumbing; quality metrics "
            "need a true BF16 GGUF."
        )

    return FreezeResult(
        model_ref=model_ref,
        method=method,
        gguf_path=str(gguf_path.resolve()),
        gguf_sha256=sha,
        gguf_nbytes=int(info["file_size_bytes"]),
        is_bf16_reference=is_bf16_ref,
        source_path=str(src.resolve()) if src and src.is_file() else (
            str(hf_dir.resolve()) if hf_dir else None
        ),
        source_is_quantized=bool(source_is_quantized) and not is_bf16_ref,
        dtype_summary=dtype_summary,
        n_tensors=int(info["n_tensors"]),
        catalog_match=catalog_match,
        catalog_missing=missing[:50],
        steps_log=log,
        notes=notes,
    )
