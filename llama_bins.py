"""Shared helpers to locate llama.cpp binaries."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def find_llama_binary(name: str, explicit: str | Path | None = None) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.is_file() else None
    which = shutil.which(name)
    if which:
        return Path(which)
    env = os.environ.get("LLAMA_CPP_DIR") or os.environ.get("LLAMA_CPP")
    candidates: list[Path] = []
    if env:
        root = Path(env).expanduser()
        candidates += [
            root / name,
            root / "build" / "bin" / name,
            root / "bin" / name,
        ]
    home = Path.home()
    candidates += [
        home / "llama.cpp" / "build" / "bin" / name,
        home / "llama.cpp" / name,
        Path(f"/opt/homebrew/bin/{name}"),
        Path(f"/usr/local/bin/{name}"),
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None
