"""Known mappings from Ollama / MLX names → original Hugging Face BF16 repos."""

from __future__ import annotations

# Ollama library name (without :tag) → HF repo that holds full-precision weights.
# Keep this small and explicit; heuristics fill gaps.
OLLAMA_TO_HF: dict[str, str] = {
    "functiongemma": "google/functiongemma-270m-it",
    "gemma3": "google/gemma-3-270m-it",
    "gemma2": "google/gemma-2-2b-it",
    "gemma": "google/gemma-2-2b-it",
    "llama3.2": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.1": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-coder": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "phi4": "microsoft/phi-4",
}

# MLX-style ids / tags → HF BF16 source (never use the MLX quantized weights).
MLX_TO_HF: dict[str, str] = {
    "gemma4:e2b-mlx": "google/gemma-3-4b-it",  # placeholder family mapping; refine when gemma4 ships
    "mlx-community/gemma-2-2b-it-4bit": "google/gemma-2-2b-it",
}


def ollama_name_from_tag(tag: str) -> str:
    """functiongemma:latest → functiongemma"""
    return tag.split(":", 1)[0].strip().lower()


def lookup_ollama_hf(tag: str) -> str | None:
    name = ollama_name_from_tag(tag)
    if name in OLLAMA_TO_HF:
        return OLLAMA_TO_HF[name]
    # Heuristic: functiongemma → google/functiongemma-270m-it style already covered;
    # try google/<name> for gemma-family names.
    if "gemma" in name or name.startswith("function"):
        # Prefer instruct / it suffix when unknown size.
        return f"google/{name}-270m-it" if "270" not in name and "functiongemma" in name else f"google/{name}"
    return None


def lookup_mlx_hf(ref: str) -> str | None:
    key = ref.strip().lower()
    if key in MLX_TO_HF:
        return MLX_TO_HF[key]
    # Strip -mlx / :mlx suffixes and hope for a known ollama/hf name.
    bare = key.replace(":mlx", "").replace("-mlx", "").removesuffix(":latest")
    return lookup_ollama_hf(bare)
