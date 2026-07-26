# Step 02 — Load the model (implementation)

Uses Step 01 checkpoint `local_path`.

## Ollama / GGUF (default)

```bash
odg load --model functiongemma:latest
```

What happens:

1. Read `steps/01_resolve/output.json`
2. Open the Ollama GGUF blob
3. Parse metadata + **full tensor index** (names, shapes, dtypes)
4. Checkpoint to `steps/02_load/`:
   - `output.json` — summary
   - `tensor_index.json` — all tensors (for Step 03)
   - `log.txt`, `status.json`, `input.json`

Weights are **not** dequantized into BF16 RAM — only the header/index is loaded. That matches the current “use Ollama” workflow.

## HF safetensors (later)

After `odg resolve --prefer-hf --download-weights`, Step 02 indexes `*.safetensors` via mmap.

## Code

| File | Role |
|---|---|
| `odg/load/load.py` | Orchestrator |
| `odg/load/gguf_load.py` | GGUF header + tensor table |
| `odg/load/hf_load.py` | HF directory / safetensors |
| `odg/load/types.py` | `LoadedModel` |
