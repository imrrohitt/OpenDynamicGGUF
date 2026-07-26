"""Hugging Face Hub helpers for Step 01 (config fetch / optional weight download)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import ArchitectureDescriptor


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
