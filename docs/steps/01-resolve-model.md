# Step 01 — Resolve model to original BF16

← [Index](./README.md) · Next: [02 Load model](./02-load-model.md) →

---

## Goal

Take **any** way a user names a model and resolve it to the **original full-precision (BF16/F16) Hugging Face safetensors checkpoint**.

Never start quantization from an already-quantized artifact.

---

## Why it exists

Quantization error **stacks**. If you take an MLX 4-bit or Ollama Q4 GGUF and quantize again, quality collapses and no later probe can fix it.

```text
BAD:  gemma4:e2b-mlx (already 4-bit) → quantize again → garbage
GOOD: gemma4:e2b-mlx → find original HF BF16 → use that
```

---

## Inputs / Outputs

| | |
|---|---|
| **Input** | User string: Ollama tag, MLX id, HF repo, or local path |
| **Output** | Local path (or HF id) to BF16 safetensors + architecture descriptor |

**Architecture descriptor** (example fields):

```yaml
family: gemma
layer_count: 18
is_moe: false
is_hybrid_ssm: false
chat_template: gemma
specialty_domain: null   # or "function_calling" for FunctionGemma
source_sha256: "…"
```

---

## How it works

```text
User input
    │
    ├─ looks like HF repo?     → use directly (if full precision)
    ├─ looks like local dir?   → check for safetensors + config.json
    ├─ looks like Ollama tag?  → read Ollama manifest → find upstream HF
    └─ looks like MLX repo?    → reject as source → map to original HF
    │
    ▼
Download / verify BF16 weights
    │
    ▼
Write architecture descriptor
```

### Decision table

| You give it | What it is | Resolver action |
|---|---|---|
| `google/gemma-3-270m` | HF BF16 | Use directly |
| `./my-finetune/` | Local full-precision | Use directly |
| `functiongemma:latest` | Ollama → often a Q4 GGUF | Manifest → upstream HF fine-tune |
| `gemma4:e2b-mlx` | MLX quantized weights | **Not a source** → original HF BF16 |

---

## Example

```text
User:  odg quantize --model functiongemma:latest --target-size 3.2GB

Resolver:
  1. Detect Ollama-style tag
  2. Read local/remote Ollama manifest
  3. Find upstream: google/functiongemma-… (illustrative)
  4. Confirm weights are BF16/F16 safetensors (not Q4)
  5. Emit:
       model_path = ~/.cache/odg/models/functiongemma-bf16/
       descriptor = { family: gemma, specialty_domain: function_calling, … }
```

Pseudo-code:

```python
def resolve(user_ref: str) -> ResolvedModel:
    kind = classify_ref(user_ref)  # hf | ollama | mlx | local

    if kind == "mlx":
        user_ref = map_mlx_to_hf(user_ref)   # never use MLX weights
        kind = "hf"

    if kind == "ollama":
        user_ref = ollama_upstream_hf(user_ref)
        kind = "hf"

    path = ensure_bf16_safetensors(user_ref)
    assert not looks_quantized(path)
    return ResolvedModel(path=path, descriptor=inspect_config(path))
```

---

## Done when

- [x] You have identified the **full-precision** upstream (`google/functiongemma-270m-it` for `functiongemma:latest`)
- [x] Descriptor has family, layer count, chat template, specialty domain (from Ollama/GGUF metadata; HF config when unlocked)
- [ ] Source BF16 weights downloaded (requires Hugging Face login for gated Gemma repos)

## Implementation

See [01-resolve-model-impl.md](./01-resolve-model-impl.md) and package `odg/resolve/`.

```bash
source .venv/bin/activate
odg resolve --model functiongemma:latest --out artifacts/resolve-functiongemma.json
```

## Next

[Step 02 — Load the model into memory](./02-load-model.md)
