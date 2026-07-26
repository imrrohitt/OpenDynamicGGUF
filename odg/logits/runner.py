"""Locate and run llama-perplexity for KL base caches."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def find_llama_perplexity(explicit: str | Path | None = None) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.is_file() else None
    which = shutil.which("llama-perplexity")
    if which:
        return Path(which)
    env = os.environ.get("LLAMA_CPP_DIR") or os.environ.get("LLAMA_CPP")
    candidates: list[Path] = []
    if env:
        root = Path(env).expanduser()
        candidates += [
            root / "llama-perplexity",
            root / "build" / "bin" / "llama-perplexity",
            root / "bin" / "llama-perplexity",
        ]
    home = Path.home()
    candidates += [
        home / "llama.cpp" / "build" / "bin" / "llama-perplexity",
        home / "llama.cpp" / "llama-perplexity",
        Path("/opt/homebrew/bin/llama-perplexity"),
        Path("/usr/local/bin/llama-perplexity"),
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def run_kl_divergence_base(
    *,
    binary: Path,
    model_gguf: Path,
    text_file: Path,
    outfile: Path,
    extra_args: list[str] | None = None,
) -> str:
    """
    Cache reference logits:
      llama-perplexity -m MODEL -f TEXT --kl-divergence-base OUT
    """
    outfile.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(binary),
        "-m",
        str(model_gguf),
        "-f",
        str(text_file),
        "--kl-divergence-base",
        str(outfile),
    ]
    if extra_args:
        cmd += list(extra_args)

    proc = subprocess.run(cmd, capture_output=True, text=True)
    log = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        raise RuntimeError(
            f"llama-perplexity failed (exit {proc.returncode}):\n"
            f"cmd: {' '.join(cmd)}\n"
            f"{log[-4000:]}"
        )
    if not outfile.is_file():
        raise RuntimeError(
            f"llama-perplexity exited 0 but outfile missing: {outfile}\n{log[-2000:]}"
        )
    return log
