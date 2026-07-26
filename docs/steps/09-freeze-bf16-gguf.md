# Step 09 — Freeze the BF16 GGUF reference

← [08 Activations](./08-compute-activation-features.md) · [Index](./README.md) · Next: [10 Imatrix](./10-build-imatrix.md) →

---

## Goal

Convert the same HF BF16 checkpoint into a **frozen, hashed BF16 GGUF**. All llama.cpp probes, imatrix, and exports use this file.

---

## Why it exists

Two views of the **same** weights:

| View | Tooling | Purpose |
|---|---|---|
| HF / PyTorch | Steps 02–08 | Catalog, weight/activation features |
| BF16 GGUF | Steps 09–15 | imatrix, trial quant, KLD, final export |

If these diverge, measurements become meaningless. Hash the GGUF and record it everywhere.

---

## Inputs / Outputs

| | |
|---|---|
| **Input** | Same BF16 HF path as Step 01 |
| **Output** | `model-bf16.gguf` + SHA-256 |

---

## How it works

```bash
python llama.cpp/convert_hf_to_gguf.py \
  ./model-src \
  --outtype bf16 \
  --outfile model-bf16.gguf

sha256sum model-bf16.gguf > model-bf16.gguf.sha256
```

Verify catalog GGUF names match tensors inside the file (optional `llama-gguf` / dump tool).

---

## Example

```text
Input:  ~/.cache/odg/models/gemma-3-270m/   (safetensors)
Output: artifacts/model-bf16.gguf
Size:   ~540 MB (bf16 for ~270M params)
SHA:    9f2c…
```

Recipe / cache key later includes this SHA so results are reproducible.

---

## Done when

- [ ] BF16 GGUF exists and loads in llama.cpp
- [ ] SHA recorded next to catalog `source_sha256`
- [ ] Confirmed not accidentally F32-duplicated or already quantized

## Next

[Step 10 — Build the importance matrix](./10-build-imatrix.md)
