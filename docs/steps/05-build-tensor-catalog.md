# Step 05 — Build the tensor catalog

← [04 Classify](./04-classify-tensors.md) · [Index](./README.md) · Next: [06 Weight features](./06-compute-weight-features.md) →

---

## Goal

Write `tensor_catalog.json` — the durable inventory used by every later step.

---

## Command

```bash
odg catalog --model functiongemma:latest
odg status --model functiongemma:latest
```

Requires Step 04 done.

---

## Outputs

```text
steps/05_catalog/
  output.json           # summary + sha256
  tensor_catalog.json   # full catalog
  status.json
  log.txt
```

Each tensor entry includes: `gguf_name`, `hf_name`, role, depth, `group_id`,
`quantizable`, and null slots for `weight_features` / `activation_features`.

---

## Done when

- [x] Catalog written and hashed (`catalog_sha256`)
- [x] Groups listed
- [x] GGUF names present (primary keys when source is Ollama)

## Next

[Step 06 — Compute weight features](./06-compute-weight-features.md)
