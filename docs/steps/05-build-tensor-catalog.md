# Step 05 — Build the tensor catalog

← [04 Classify](./04-classify-tensors.md) · [Index](./README.md) · Next: [06 Weight features](./06-compute-weight-features.md) →

---

## Goal

Persist a single structured inventory of every tensor: HF name, GGUF name (when known), role, shape, and flags. This file is the **source of truth** for later probing and export.

---

## Why it exists

Probes talk to llama.cpp using GGUF names (`blk.12.attn_v.weight`). Analysis talks in HF names (`…v_proj.weight`). The catalog is the bridge.

---

## Inputs / Outputs

| | |
|---|---|
| **Input** | Enumerated + classified tensors (Steps 03–04) |
| **Output** | `tensor_catalog.json` (content-addressed) |

---

## How it works

For each tensor store:

```text
name          HF parameter name
gguf_name     mapped llama.cpp name (or null until converter mapping known)
shape
dtype
role / layer / depth
nbytes
quantizable
group_id      e.g. "attn_v@middle"
```

### HF → GGUF name mapping (illustrative)

| HF | GGUF |
|---|---|
| `model.layers.12.self_attn.q_proj.weight` | `blk.12.attn_q.weight` |
| `model.layers.12.mlp.down_proj.weight` | `blk.12.ffn_down.weight` |
| `model.embed_tokens.weight` | `token_embd.weight` |
| `lm_head.weight` | `output.weight` |

Mapping rules live in `catalog.py` and must match `convert_hf_to_gguf.py` for the family.

### Groups

```python
group_id = f"{role}@{depth}" if depth else role
# Examples: "ffn_up@middle", "attn_v@all", "embedding"
```

Later, Step 12 probes **groups**, not every single tensor.

---

## Example catalog entry

```json
{
  "model_source": "google/gemma-3-270m",
  "source_sha256": "abc123…",
  "n_layers": 18,
  "tensors": {
    "model.layers.12.self_attn.v_proj.weight": {
      "shape": [640, 640],
      "dtype": "torch.bfloat16",
      "role": "attn_v",
      "layer": 12,
      "depth": "middle",
      "group_id": "attn_v@middle",
      "gguf_name": "blk.12.attn_v.weight",
      "nbytes": 819200,
      "quantizable": true,
      "weight_features": null,
      "activation_features": null
    }
  },
  "groups": {
    "attn_v@middle": {
      "role": "attn_v",
      "depth": "middle",
      "tensor_names": ["…layers.6…", "…layers.7…", "…"]
    }
  }
}
```

---

## Done when

- [ ] Catalog written and hashed
- [ ] Groups listed
- [ ] GGUF names filled for the target family (or marked TODO)

## Next

[Step 06 — Compute weight features](./06-compute-weight-features.md)
