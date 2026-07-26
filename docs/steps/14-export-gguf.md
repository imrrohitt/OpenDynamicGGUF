# Step 14 — Export the quantized GGUF from a recipe

← [13 Optimize](./13-optimize-recipe.md) · [Index](./README.md) · Next: [15 Validate](./15-validate-and-release.md) →

---

## Goal

Render `recipe.yaml` into `llama-quantize` arguments (`--tensor-type-file`, embedding/output types) and produce the candidate GGUF with provenance metadata.

---

## Why it exists

Reproducibility is the product. Anyone with the recipe + BF16 source + imatrix should rebuild the **same** file.

---

## Inputs / Outputs

| | |
|---|---|
| **Input** | Recipe + `model-bf16.gguf` + `imatrix.gguf` |
| **Output** | `model-UD.gguf` (+ embedded KV overrides) |

---

## How it works

1. Convert recipe overrides → `recipe.tt` (tensor-type file).  
2. Run quantize with imatrix.  
3. Stamp provenance (`--override-kv`).

```bash
./llama-quantize \
  --imatrix imatrix.gguf \
  --tensor-type-file recipe.tt \
  --token-embedding-type q8_0 \
  --output-tensor-type q8_0 \
  model-bf16.gguf model-UD.gguf q4_k_m
```

Example `recipe.tt` lines:

```text
token_embd\.weight=q8_0
output\.weight=q8_0
blk\.\d+\.attn_v\.weight=q6_k
blk\.(0|1|2|3|4|5)\.attn_q\.weight=q5_k
blk\.(1[0-9])\.ffn_up\.weight=q3_k
```

(Exact syntax follows your llama.cpp version.)

---

## Example end-to-end user command

```bash
odg quantize --model functiongemma:latest --target-size 3.2GB

→ functiongemma-UD.gguf
→ recipe.yaml
→ (pending) report.html from Step 15
```

---

## Done when

- [ ] GGUF loads in llama.cpp / Ollama / LM Studio
- [ ] File size ≈ recipe estimate
- [ ] Recipe SHA + bf16 SHA + imatrix SHA recorded

## Next

[Step 15 — Validate and release](./15-validate-and-release.md)
