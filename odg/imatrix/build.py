"""
Step 10 — Build importance matrix from calib.txt + frozen GGUF.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

from .proxy import build_proxy_importance
from .runner import find_llama_imatrix, run_llama_imatrix
from .types import ImatrixResult

Mode = Literal["auto", "llama", "proxy"]


def _sha256_file(path: Path, *, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def build_imatrix(
    *,
    model_ref: str,
    out_dir: Path,
    gguf_path: str | Path,
    calib_path: str | Path,
    catalog: dict[str, Any] | None = None,
    gguf_sha256: str | None = None,
    mode: Mode = "auto",
    llama_imatrix: str | Path | None = None,
    n_chunks: int | None = 64,
) -> ImatrixResult:
    """
    Produce imatrix.gguf (real) and/or imatrix_proxy.json under out_dir.
    """
    log: list[str] = []
    notes: list[str] = []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gguf = Path(gguf_path).expanduser().resolve()
    calib = Path(calib_path).expanduser().resolve()
    if not gguf.is_file():
        raise FileNotFoundError(f"Frozen GGUF not found: {gguf}")
    if not calib.is_file():
        raise FileNotFoundError(f"calib.txt not found: {calib}")

    if gguf_sha256 is None:
        gguf_sha256 = _sha256_file(gguf)
    log.append(f"1. GGUF: {gguf} sha256={gguf_sha256[:16]}…")
    log.append(f"2. Calib: {calib} (must NOT be heldout)")

    binary = find_llama_imatrix(llama_imatrix)
    want_llama = mode in {"auto", "llama"}

    imatrix_path: Path | None = None
    imatrix_sha: str | None = None
    proxy_path: Path | None = None
    method: str
    n_scored = 0

    if want_llama and binary is not None:
        imatrix_path = out_dir / "imatrix.gguf"
        log.append(f"3. Running llama-imatrix: {binary}")
        if n_chunks:
            log.append(f"4. chunks={n_chunks}")
        try:
            run_log = run_llama_imatrix(
                binary=binary,
                model_gguf=gguf,
                calib_txt=calib,
                outfile=imatrix_path,
                n_chunks=n_chunks,
            )
            (out_dir / "llama-imatrix.log").write_text(run_log, encoding="utf-8")
            imatrix_sha = _sha256_file(imatrix_path)
            method = "llama_imatrix"
            log.append(f"5. Wrote {imatrix_path} sha256={imatrix_sha[:16]}…")
            notes.append("Real imatrix.gguf ready for llama-quantize --imatrix.")
            # Also write proxy ranking for our optimizer readability
            if catalog:
                proxy_path = out_dir / "imatrix_proxy.json"
                payload = build_proxy_importance(
                    catalog,
                    out_path=proxy_path,
                    gguf_sha256=gguf_sha256,
                    calib_path=str(calib),
                )
                n_scored = int(payload.get("n_tensors") or 0)
                log.append(f"6. Also wrote readable proxy ranking ({n_scored} tensors)")
        except Exception:
            if mode == "llama":
                raise
            log.append("5. llama-imatrix failed — falling back to proxy_importance")
            imatrix_path = None
            method = "proxy_importance"
            if not catalog:
                raise RuntimeError(
                    "llama-imatrix failed and no catalog for proxy fallback"
                )
            proxy_path = out_dir / "imatrix_proxy.json"
            payload = build_proxy_importance(
                catalog,
                out_path=proxy_path,
                gguf_sha256=gguf_sha256,
                calib_path=str(calib),
            )
            n_scored = int(payload.get("n_tensors") or 0)
            notes.append(
                "llama-imatrix failed; proxy_importance written. "
                "Not usable as --imatrix until real binary succeeds."
            )
    elif mode == "llama":
        raise RuntimeError(
            "llama-imatrix not found. Install llama.cpp and ensure llama-imatrix "
            "is on PATH, or set LLAMA_CPP_DIR / --llama-imatrix."
        )
    else:
        method = "proxy_importance"
        if binary is None:
            log.append("3. llama-imatrix not found — using proxy_importance")
        else:
            log.append("3. mode=proxy — skipping llama-imatrix")
        if not catalog:
            raise RuntimeError("catalog required for proxy_importance mode")
        proxy_path = out_dir / "imatrix_proxy.json"
        payload = build_proxy_importance(
            catalog,
            out_path=proxy_path,
            gguf_sha256=gguf_sha256,
            calib_path=str(calib),
        )
        n_scored = int(payload.get("n_tensors") or 0)
        log.append(f"4. Wrote {proxy_path} ({n_scored} tensors scored)")
        notes.append(
            "Proxy only — install llama-imatrix and re-run with --mode llama "
            "for a real imatrix.gguf."
        )
        # Marker so later steps know what's missing
        (out_dir / "imatrix.gguf.MISSING").write_text(
            "Run: odg imatrix --mode llama --force\n", encoding="utf-8"
        )

    return ImatrixResult(
        model_ref=model_ref,
        method=method,
        gguf_path=str(gguf),
        gguf_sha256=gguf_sha256,
        calib_path=str(calib),
        imatrix_path=str(imatrix_path) if imatrix_path and imatrix_path.is_file() else None,
        imatrix_sha256=imatrix_sha,
        proxy_path=str(proxy_path) if proxy_path and proxy_path.is_file() else None,
        n_chunks=n_chunks if method == "llama_imatrix" else None,
        n_tensors_scored=n_scored,
        steps_log=log,
        notes=notes,
    )
