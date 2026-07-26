# Step 03 — Enumerate every tensor

← [02 Load](./02-load-model.md) · [Index](./README.md) · Next: [04 Classify](./04-classify-tensors.md) →

---

## Goal

Produce a clean flat inventory of **every** tensor: name, shape, dtype, nbytes, layer.

You know *what exists* — not yet what role it plays (Step 04).

---

## Inputs / Outputs

| | |
|---|---|
| **Input** | `02_load/tensor_index.json` |
| **Output** | `03_enumerate/output.json` (summary) + `tensors.json` + `tensors.tsv` |

---

## Command

```bash
odg enumerate --model functiongemma:latest
odg status --model functiongemma:latest
```

Requires Step 02 done.

---

## Example (FunctionGemma from Ollama)

```text
n_tensors      = 236
dtype_summary  = {Q8_0: 127, F32: 109}
layers         = 0..17 (+ global: token_embd, output_norm, …)

blk.0.attn_q.weight     1024×640   Q8_0
blk.0.ffn_up.weight     2048×640   Q8_0
token_embd.weight     262144×640   Q8_0
```

---

## Done when

- [x] Every tensor from Step 02 listed
- [x] nbytes + layer id filled
- [x] Checkpoint under `steps/03_enumerate/`

## Next

[Step 04 — Classify each tensor by role](./04-classify-tensors.md)
