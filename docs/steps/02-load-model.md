# Step 02 — Load the BF16 model into memory

← [01 Resolve](./01-resolve-model.md) · [Index](./README.md) · Next: [03 Enumerate](./03-enumerate-tensors.md) →

---

## Goal

Load the resolved checkpoint so every parameter tensor is available in memory (or memory-mapped) for inspection.

---

## Why it exists

Later steps need to **walk every weight**, classify it, and compute statistics. That requires a real `nn.Module` (or equivalent safetensors iterator), not just a GGUF file.

---

## Inputs / Outputs

| | |
|---|---|
| **Input** | Path / HF id from Step 01 |
| **Output** | Loaded `model` object in BF16 |

---

## How it works

```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,                  # from resolver — never an MLX/Ollama quant
    torch_dtype=torch.bfloat16,
    device_map="cpu",            # catalog/features can stay on CPU
)
model.eval()
```

Notes:

- Prefer **CPU** (or disk offload) for cataloging large models — you are not training.
- Keep `torch_dtype=bfloat16` (or float16) so you are inspecting the true full-precision weights.
- Do **not** load 4-bit / 8-bit bitsandbytes weights here.

---

## Example

```text
MODEL_PATH = ~/.cache/odg/models/gemma-3-270m/

Loaded:
  GemmaForCausalLM
  layers: 18
  hidden_size: 640
  vocab_size: 262144
```

For very large models:

```python
# Option A: safetensors only (no full module) — still enough for weight stats
from safetensors import safe_open
with safe_open("model.safetensors", framework="pt") as f:
    for key in f.keys():
        t = f.get_tensor(key)
        ...

# Option B: device_map="auto" / low_cpu_mem_usage=True when needed
```

---

## Done when

- [ ] Model (or safetensors handle) opens without loading a quantized format
- [ ] `config.json` readable (layer count, architectures)

## Next

[Step 03 — Enumerate every tensor](./03-enumerate-tensors.md)
