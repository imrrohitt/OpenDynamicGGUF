# Step 04 — Classify tensors by role

← [03 Enumerate](./03-enumerate-tensors.md) · [Index](./README.md) · Next: [05 Catalog](./05-build-tensor-catalog.md) →

---

## Goal

Map every tensor to a **role**, **depth** bucket, **group_id**, and **quantizable** flag.

---

## Command

```bash
odg classify --model functiongemma:latest
odg status --model functiongemma:latest
```

Requires Step 03 done.

---

## Outputs

```text
steps/04_classify/
  output.json        # summary
  classified.json    # every tensor + role/depth/group
  classified.tsv
  status.json
  log.txt
```

---

## Example (FunctionGemma GGUF)

| Role | Count | Quantizable? |
|---|---|---|
| attn_q / k / v / o | 18 each | yes |
| ffn_gate / up / down | 18 each | yes |
| norm (attn_norm, ffn_norm, …) | many | **no** |
| embedding (`token_embd`) | 1 | yes (pin later) |

`attn_k_norm` → **norm** (not attn_k) — rule order matters.

---

## Done when

- [x] ≥95% coverage (non-`other`)
- [x] Norms `quantizable=false`
- [x] `group_id = role@depth` ready for probes

## Next

[Step 05 — Build the tensor catalog](./05-build-tensor-catalog.md)
