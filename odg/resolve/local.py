"""Local directory resolution (safetensors + config.json)."""

from __future__ import annotations

import json
from pathlib import Path

from .types import ArchitectureDescriptor


def inspect_local_dir(path: Path) -> tuple[ArchitectureDescriptor, bool]:
    """
    Validate a local checkpoint directory.

    Returns (descriptor, looks_full_precision).
    """
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Local path is not a directory: {path}")

    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"No config.json in {path}")

    config = json.loads(config_path.read_text())
    safes = list(path.glob("*.safetensors")) + list(path.glob("model*.safetensors"))
    bins = list(path.glob("pytorch_model*.bin"))
    if not safes and not bins:
        raise FileNotFoundError(
            f"No safetensors / pytorch_model*.bin weights in {path}"
        )

    # Heuristic: bitsandbytes / quantized configs
    quant_cfg = config.get("quantization_config")
    looks_quantized = quant_cfg is not None

    arch_list = config.get("architectures") or []
    arch0 = arch_list[0] if arch_list else str(config.get("model_type", "unknown"))
    family = str(config.get("model_type") or arch0).lower()

    desc = ArchitectureDescriptor(
        family=family,
        layer_count=config.get("num_hidden_layers"),
        embedding_length=config.get("hidden_size"),
        context_length=config.get("max_position_embeddings"),
        is_moe=bool(config.get("num_local_experts") or config.get("num_experts")),
        chat_template=_guess_chat_template(path, family),
        specialty_domain=_guess_specialty(path, family),
    )
    if looks_quantized:
        desc.notes.append(
            "config.json contains quantization_config — this directory may not be BF16."
        )
    return desc, not looks_quantized


def _guess_chat_template(path: Path, family: str) -> str | None:
    tok = path / "tokenizer_config.json"
    if tok.is_file():
        data = json.loads(tok.read_text())
        if data.get("chat_template"):
            return family
    return family or None


def _guess_specialty(path: Path, family: str) -> str | None:
    name = path.name.lower()
    if "function" in name or "tool" in name:
        return "function_calling"
    return None
