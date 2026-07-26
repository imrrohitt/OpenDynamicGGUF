# Step 01 — Implementation notes

Code lives in `odg/resolve/`.

```bash
# from repo root
python3 -m pip install -e .

# resolve your local Ollama model (no weight download)
odg resolve --model functiongemma:latest \
  --out artifacts/resolve-functiongemma.json
```

## What the code does for `functiongemma:latest`

1. **Classify** → `ollama`
2. **`ollama show` + local manifest** → architecture `gemma3`, quantization `Q8_0`
3. **Reject** the Ollama GGUF blob as a quantization source (already Q8_0)
4. **Map** → `google/functiongemma-270m-it` (full-precision upstream)
5. **Enrich descriptor** from GGUF *metadata* only (layer count, embed size) — never reuses quantized weights
6. **Optionally** download BF16 safetensors with `--download-weights` (needs HF login; repo is gated)

## Modules

| File | Role |
|---|---|
| `classify.py` | HF / Ollama / MLX / local |
| `ollama.py` | Inspect local Ollama + GGUF metadata |
| `maps.py` | Known Ollama/MLX → HF maps |
| `hf.py` | Hub config / snapshot download |
| `local.py` | Local safetensors dirs |
| `resolve.py` | Orchestrator |
| `types.py` | `ResolvedModel`, `ArchitectureDescriptor` |
