"""Locate llama-imatrix and run it."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def find_llama_imatrix(explicit: str | Path | None = None) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.is_file() and os.access(p, os.X_OK) else (
            p if p.is_file() else None
        )
    which = shutil.which("llama-imatrix")
    if which:
        return Path(which)
    env = os.environ.get("LLAMA_CPP_DIR") or os.environ.get("LLAMA_CPP")
    candidates: list[Path] = []
    if env:
        root = Path(env).expanduser()
        candidates += [
            root / "llama-imatrix",
            root / "build" / "bin" / "llama-imatrix",
            root / "bin" / "llama-imatrix",
        ]
    home = Path.home()
    candidates += [
        home / "llama.cpp" / "build" / "bin" / "llama-imatrix",
        home / "llama.cpp" / "llama-imatrix",
        Path("/opt/homebrew/bin/llama-imatrix"),
        Path("/usr/local/bin/llama-imatrix"),
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def run_llama_imatrix(
    *,
    binary: Path,
    model_gguf: Path,
    calib_txt: Path,
    outfile: Path,
    n_chunks: int | None = None,
    extra_args: list[str] | None = None,
) -> str:
    """
    Run llama-imatrix. Returns combined stdout+stderr tail for the log.
    """
    outfile.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(binary),
        "-m",
        str(model_gguf),
        "-f",
        str(calib_txt),
        "-o",
        str(outfile),
    ]
    if n_chunks is not None and n_chunks > 0:
        # Flag name varies slightly by version; --chunks is common
        cmd += ["--chunks", str(n_chunks)]
    if extra_args:
        cmd += list(extra_args)

    proc = subprocess.run(cmd, capture_output=True, text=True)
    log = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        raise RuntimeError(
            f"llama-imatrix failed (exit {proc.returncode}):\n"
            f"cmd: {' '.join(cmd)}\n"
            f"{log[-4000:]}"
        )
    if not outfile.is_file():
        raise RuntimeError(
            f"llama-imatrix exited 0 but outfile missing: {outfile}\n{log[-2000:]}"
        )
    return log
