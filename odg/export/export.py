"""
Step 14 — Export quantized GGUF from recipe via llama-quantize (or dry-run).
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal

from odg.llama_bins import find_llama_binary

from .types import ExportResult

Mode = Literal["auto", "llama", "dry-run"]


def _sha256_file(path: Path, *, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _parse_recipe_estimate(recipe_path: Path) -> int | None:
    text = recipe_path.read_text(encoding="utf-8")
    m = re.search(r"size_bytes:\s*(\d+)", text)
    return int(m.group(1)) if m else None


def _embedding_output_types(recipe_tt: Path) -> tuple[str | None, str | None]:
    emb, out = None, None
    for line in recipe_tt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        pat, q = line.split("=", 1)
        q = q.strip()
        # Exact-ish: token_embd / output head — never attn_output
        if "token_embd" in pat or "embed_tokens" in pat:
            emb = q
        if re.search(r"(^|[^\w])output\.weight", pat) or pat.strip() in {
            "output.weight",
            "output",
        }:
            out = q
    return emb, out


def build_quantize_command(
    *,
    binary: Path,
    gguf_in: Path,
    gguf_out: Path,
    recipe_tt: Path,
    imatrix: Path | None,
    base_type: str = "q4_k_m",
    embedding_type: str | None = None,
    output_type: str | None = None,
) -> list[str]:
    cmd = [str(binary)]
    if imatrix and imatrix.is_file():
        cmd += ["--imatrix", str(imatrix)]
    cmd += ["--tensor-type-file", str(recipe_tt)]
    if embedding_type:
        cmd += ["--token-embedding-type", embedding_type]
    if output_type:
        cmd += ["--output-tensor-type", output_type]
    # Provenance KV (best-effort; ignored if unsupported)
    cmd += [
        "--override-kv",
        "general.description=str:OpenDynamicGGUF dynamic quant",
    ]
    cmd += [str(gguf_in), str(gguf_out), base_type]
    return cmd


def export_gguf(
    *,
    model_ref: str,
    out_dir: Path,
    gguf_in: str | Path,
    recipe_path: str | Path,
    recipe_tt: str | Path,
    imatrix_path: str | Path | None = None,
    mode: Mode = "auto",
    llama_quantize: str | Path | None = None,
    base_type: str = "q4_k_m",
    out_name: str | None = None,
) -> ExportResult:
    log: list[str] = []
    notes: list[str] = []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gguf_in = Path(gguf_in).expanduser().resolve()
    recipe_path = Path(recipe_path).expanduser().resolve()
    recipe_tt = Path(recipe_tt).expanduser().resolve()
    imatrix = Path(imatrix_path).expanduser().resolve() if imatrix_path else None

    if not gguf_in.is_file():
        raise FileNotFoundError(f"Input GGUF not found: {gguf_in}")
    if not recipe_path.is_file():
        raise FileNotFoundError(f"recipe.yaml not found: {recipe_path}")
    if not recipe_tt.is_file():
        raise FileNotFoundError(f"recipe.tt not found: {recipe_tt}")

    # Copy recipe artifacts into export step for provenance
    shutil.copy2(recipe_path, out_dir / "recipe.yaml")
    shutil.copy2(recipe_tt, out_dir / "recipe.tt")

    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", model_ref).strip("-") or "model"
    out_name = out_name or f"{slug}-UD.gguf"
    gguf_out = out_dir / out_name

    emb, out_t = _embedding_output_types(recipe_tt)
    estimated = _parse_recipe_estimate(recipe_path)

    binary = find_llama_binary("llama-quantize", llama_quantize)
    want = mode in {"auto", "llama"}
    method: str
    command: list[str] = []
    out_sha: str | None = None
    out_nbytes: int | None = None
    produced: str | None = None

    log.append(f"1. Input GGUF: {gguf_in}")
    log.append(f"2. Recipe: {recipe_path.name} + {recipe_tt.name}")
    if imatrix and imatrix.is_file():
        log.append(f"3. Imatrix: {imatrix}")
    else:
        log.append("3. Imatrix: (none / proxy) — quantize may still run without it")
        imatrix = None
        notes.append("No real imatrix.gguf — export without --imatrix if dry-run/proxy.")

    if want and binary is not None:
        command = build_quantize_command(
            binary=binary,
            gguf_in=gguf_in,
            gguf_out=gguf_out,
            recipe_tt=out_dir / "recipe.tt",
            imatrix=imatrix,
            base_type=base_type,
            embedding_type=emb,
            output_type=out_t,
        )
        log.append(f"4. Running llama-quantize: {binary}")
        try:
            proc = subprocess.run(command, capture_output=True, text=True)
            (out_dir / "llama-quantize.log").write_text(
                (proc.stdout or "") + "\n" + (proc.stderr or ""),
                encoding="utf-8",
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"llama-quantize exit {proc.returncode}:\n"
                    f"{(proc.stderr or proc.stdout or '')[-3000:]}"
                )
            if not gguf_out.is_file():
                raise RuntimeError("llama-quantize succeeded but output missing")
            method = "llama_quantize"
            out_sha = _sha256_file(gguf_out)
            out_nbytes = gguf_out.stat().st_size
            produced = str(gguf_out)
            log.append(
                f"5. Wrote {gguf_out.name} "
                f"({out_nbytes / (1024**2):.1f} MiB) sha={out_sha[:16]}…"
            )
            notes.append("Candidate GGUF ready for Step 15 validation.")
        except Exception:
            if mode == "llama":
                raise
            log.append("5. llama-quantize failed — falling back to dry_run")
            method = "dry_run"
            command = build_quantize_command(
                binary=binary,
                gguf_in=gguf_in,
                gguf_out=gguf_out,
                recipe_tt=out_dir / "recipe.tt",
                imatrix=imatrix,
                base_type=base_type,
                embedding_type=emb,
                output_type=out_t,
            )
    elif mode == "llama":
        raise RuntimeError(
            "llama-quantize not found. Install llama.cpp or set LLAMA_CPP_DIR / "
            "--llama-quantize."
        )
    else:
        method = "dry_run"
        # Still record the command that would be run
        fake_bin = binary or Path("llama-quantize")
        command = build_quantize_command(
            binary=fake_bin,
            gguf_in=gguf_in,
            gguf_out=gguf_out,
            recipe_tt=out_dir / "recipe.tt",
            imatrix=imatrix,
            base_type=base_type,
            embedding_type=emb,
            output_type=out_t,
        )
        log.append("4. dry_run — llama-quantize not available")
        (out_dir / f"{out_name}.MISSING").write_text(
            "Run with --mode llama after installing llama-quantize:\n"
            + " ".join(command)
            + "\n",
            encoding="utf-8",
        )
        notes.append(
            "Dry-run only: no GGUF produced. Install llama-quantize and "
            "re-run: odg export --mode llama --force"
        )
        log.append("5. Wrote command + MISSING marker")

    manifest: dict[str, Any] = {
        "model_ref": model_ref,
        "method": method,
        "gguf_in": str(gguf_in),
        "gguf_out": produced,
        "gguf_out_sha256": out_sha,
        "gguf_out_nbytes": out_nbytes,
        "recipe_path": str(out_dir / "recipe.yaml"),
        "tensor_type_file": str(out_dir / "recipe.tt"),
        "imatrix_path": str(imatrix) if imatrix else None,
        "command": command,
        "estimated_bytes": estimated,
        "base_type": base_type,
    }
    (out_dir / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "quantize_command.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + " ".join(shlex_quote(c) for c in command)
        + "\n",
        encoding="utf-8",
    )

    return ExportResult(
        model_ref=model_ref,
        method=method,
        gguf_in=str(gguf_in),
        gguf_out=produced,
        gguf_out_sha256=out_sha,
        gguf_out_nbytes=out_nbytes,
        recipe_path=str(out_dir / "recipe.yaml"),
        tensor_type_file=str(out_dir / "recipe.tt"),
        imatrix_path=str(imatrix) if imatrix else None,
        command=command,
        estimated_bytes=estimated,
        steps_log=log,
        notes=notes,
    )


def shlex_quote(s: str) -> str:
    import shlex

    return shlex.quote(s)
