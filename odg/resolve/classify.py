"""Classify a user model reference into hf / ollama / mlx / local."""

from __future__ import annotations

import re
from pathlib import Path

from .types import SourceKind

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
