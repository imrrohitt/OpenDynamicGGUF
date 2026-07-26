"""
Step 11 — Cache reference logits for search + heldout (never calib).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from .runner import find_llama_perplexity, run_kl_divergence_base
from .types import LogitsResult

Mode = Literal["auto", "llama", "proxy"]


def _sha256_file(path: Path, *, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _cache_key(gguf_sha: str, search_sha: str, heldout_sha: str) -> str:
    blob = f"odg-logits-v1|{gguf_sha}|{search_sha}|{heldout_sha}".encode()
    return hashlib.sha256(blob).hexdigest()


def cache_reference_logits(
    *,
    model_ref: str,
    out_dir: Path,
    gguf_path: str | Path,
    search_path: str | Path,
    heldout_path: str | Path,
    gguf_sha256: str | None = None,
    mode: Mode = "auto",
    llama_perplexity: str | Path | None = None,
) -> LogitsResult:
    log: list[str] = []
    notes: list[str] = []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gguf = Path(gguf_path).expanduser().resolve()
    search = Path(search_path).expanduser().resolve()
    heldout = Path(heldout_path).expanduser().resolve()

    for p, label in ((gguf, "GGUF"), (search, "search.txt"), (heldout, "heldout.txt")):
        if not p.is_file():
            raise FileNotFoundError(f"{label} not found: {p}")

    # Hard wall: never accept calib as search/heldout by path name confusion
    for p, name in ((search, "search"), (heldout, "heldout")):
        if p.name.startswith("calib"):
            raise ValueError(
                f"{name} path looks like calib ({p.name}) — refuse to cache"
            )

    gguf_sha = gguf_sha256 or _sha256_file(gguf)
    search_sha = _sha256_file(search)
    heldout_sha = _sha256_file(heldout)
    key = _cache_key(gguf_sha, search_sha, heldout_sha)

    log.append(f"1. GGUF sha256={gguf_sha[:16]}…")
    log.append(f"2. search sha256={search_sha[:16]}… heldout sha256={heldout_sha[:16]}…")
    log.append(f"3. cache_key={key[:24]}…")
    log.append("4. Hard rule: search for probes; heldout for validation ONLY")

    binary = find_llama_perplexity(llama_perplexity)
    want = mode in {"auto", "llama"}

    logits_search: Path | None = None
    logits_heldout: Path | None = None
    search_logits_sha: str | None = None
    heldout_logits_sha: str | None = None
    method: str

    if want and binary is not None:
        log.append(f"5. Running llama-perplexity: {binary}")
        logits_search = out_dir / "logits-search.bin"
        logits_heldout = out_dir / "logits-heldout.bin"
        try:
            log_s = run_kl_divergence_base(
                binary=binary,
                model_gguf=gguf,
                text_file=search,
                outfile=logits_search,
            )
            log_h = run_kl_divergence_base(
                binary=binary,
                model_gguf=gguf,
                text_file=heldout,
                outfile=logits_heldout,
            )
            (out_dir / "llama-perplexity-search.log").write_text(
                log_s, encoding="utf-8"
            )
            (out_dir / "llama-perplexity-heldout.log").write_text(
                log_h, encoding="utf-8"
            )
            search_logits_sha = _sha256_file(logits_search)
            heldout_logits_sha = _sha256_file(logits_heldout)
            method = "llama_perplexity"
            log.append(
                f"6. Wrote logits-search.bin ({search_logits_sha[:16]}…) "
                f"and logits-heldout.bin ({heldout_logits_sha[:16]}…)"
            )
            notes.append(
                "Real KL base caches ready for llama-perplexity --kl-divergence."
            )
        except Exception:
            if mode == "llama":
                raise
            log.append("6. llama-perplexity failed — writing proxy_manifest")
            method = "proxy_manifest"
            logits_search = None
            logits_heldout = None
    elif mode == "llama":
        raise RuntimeError(
            "llama-perplexity not found. Install llama.cpp and ensure "
            "llama-perplexity is on PATH, or set LLAMA_CPP_DIR / --llama-perplexity."
        )
    else:
        method = "proxy_manifest"
        if binary is None:
            log.append("5. llama-perplexity not found — proxy_manifest only")
        else:
            log.append("5. mode=proxy — skipping llama-perplexity")

    if method == "proxy_manifest":
        for name in ("logits-search.bin", "logits-heldout.bin"):
            (out_dir / f"{name}.MISSING").write_text(
                "Run: odg reference-logits --mode llama --force\n"
                f"Needs: llama-perplexity -m GGUF -f SPLIT "
                f"--kl-divergence-base {name}\n",
                encoding="utf-8",
            )
        notes.append(
            "Proxy only — logit .bin caches not produced. "
            "Install llama-perplexity and re-run with --mode llama."
        )
        log.append("6. Wrote MISSING markers + logits_manifest.json")

    manifest = {
        "model_ref": model_ref,
        "method": method,
        "cache_key": key,
        "gguf_path": str(gguf),
        "gguf_sha256": gguf_sha,
        "search_path": str(search),
        "heldout_path": str(heldout),
        "search_sha256": search_sha,
        "heldout_sha256": heldout_sha,
        "logits_search_path": str(logits_search)
        if logits_search and logits_search.is_file()
        else None,
        "logits_heldout_path": str(logits_heldout)
        if logits_heldout and logits_heldout.is_file()
        else None,
        "logits_search_sha256": search_logits_sha,
        "logits_heldout_sha256": heldout_logits_sha,
        "notes": notes,
        "rules": [
            "Never use calib.txt for reference logits.",
            "search → Steps 12–13 probes/optimize",
            "heldout → Step 15 validation only",
        ],
    }
    (out_dir / "logits_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    return LogitsResult(
        model_ref=model_ref,
        method=method,
        gguf_path=str(gguf),
        gguf_sha256=gguf_sha,
        search_path=str(search),
        heldout_path=str(heldout),
        search_sha256=search_sha,
        heldout_sha256=heldout_sha,
        logits_search_path=manifest["logits_search_path"],
        logits_heldout_path=manifest["logits_heldout_path"],
        logits_search_sha256=search_logits_sha,
        logits_heldout_sha256=heldout_logits_sha,
        cache_key=key,
        steps_log=log,
        notes=notes,
    )
