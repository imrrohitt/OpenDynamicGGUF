# Step 04 — Classify tensors by role

← [03 Enumerate](./03-enumerate-tensors.md) · [Index](./README.md) · Next: [05 Catalog](./05-build-tensor-catalog.md) →

---

## Goal

Map every tensor name to a **role** (and layer index / depth bucket) so the optimizer can treat attention-V differently from MLP-up, and skip norms.

---

## Why it exists

Treating all tensors identically is weak. Empirically:

- `attn_v` is often **sensitive** (needs higher bits)
- `ffn_up` / `ffn_gate` are often **cheap** to compress
- LayerNorm / RMSNorm is usually **left in F16/F32**

Classification encodes that structure **before** any probing.

---

## Inputs / Outputs

| | |
|---|---|
| **Input** | Flat tensor list from Step 03 |
| **Output** | Same list + `role`, `layer`, `depth`, `quantizable` |

---

## How it works

### Role taxonomy

| Role | Name patterns (examples) | Quantizable? |
|---|---|---|
| `embedding` | `embed_tokens`, `tok_embeddings` | Yes (often pin high) |
| `attn_q` | `q_proj`, `wq`, `attn_q` | Yes |
| `attn_k` | `k_proj`, `wk`, `attn_k` | Yes |
| `attn_v` | `v_proj`, `wv`, `attn_v` | Yes |
| `attn_o` | `o_proj`, `wo`, `attn_output` | Yes |
| `ffn_gate` | `gate_proj`, `w1`, `ffn_gate` | Yes |
| `ffn_up` | `up_proj`, `w3`, `ffn_up` | Yes |
| `ffn_down` | `down_proj`, `w2`, `ffn_down` | Yes |
| `ffn_*_exps` | `experts.*` (MoE) | Yes |
| `router` | MoE `gate` / `router` | Careful / pin |
| `ssm_*` | Hybrid / Mamba | Role-dependent |
| `norm` | `layernorm`, `rms_norm`, `norm` | **Usually skip** |
| `lm_head` | `lm_head`, `output` | Yes (often pin high) |
| `other` | unmatched | Review manually |

### Depth buckets

Parse layer index from the name (`layers.12` → `12`):

| Bucket | Example rule (18-layer model) |
|---|---|
| `early` | layers 0–5 |
| `middle` | layers 6–12 |
| `late` | layers 13–17 |

Exact boundaries can be `floor(n/3)` thirds. Role × depth ≈ **~25 groups** later.

### Classifier sketch

```python
import re

RULES = [
    (r"embed_tokens|tok_embeddings", "embedding"),
    (r"lm_head|(^|\.)output(\.|$)", "lm_head"),
    (r"(layernorm|rms_norm|(^|\.)norm)", "norm"),
    (r"q_proj|\.wq\.|attn_q", "attn_q"),
    (r"k_proj|\.wk\.|attn_k", "attn_k"),
    (r"v_proj|\.wv\.|attn_v", "attn_v"),
    (r"o_proj|\.wo\.|attn_output", "attn_o"),
    (r"gate_proj|\.w1\.|ffn_gate", "ffn_gate"),
    (r"up_proj|\.w3\.|ffn_up", "ffn_up"),
    (r"down_proj|\.w2\.|ffn_down", "ffn_down"),
]

def classify(name: str, n_layers: int) -> dict:
    role = "other"
    for pat, r in RULES:
        if re.search(pat, name, re.I):
            role = r
            break
    m = re.search(r"layers?[.\[](\d+)", name)
    layer = int(m.group(1)) if m else None
    depth = depth_bucket(layer, n_layers) if layer is not None else None
    return {
        "role": role,
        "layer": layer,
        "depth": depth,
        "quantizable": role not in ("norm",),
    }
```

---

## Example

```text
Input name:  model.layers.12.self_attn.v_proj.weight

Output:
  role:        attn_v
  layer:       12
  depth:       middle
  quantizable: true
```

```text
Input name:  model.layers.3.input_layernorm.weight

Output:
  role:        norm
  layer:       3
  depth:       early
  quantizable: false    ← skip in probes / recipe
```

---

## Done when

- [ ] ≥95% of weight tensors classified (rest flagged `other`)
- [ ] Norms marked non-quantizable
- [ ] Layer + depth present for block weights

## Next

[Step 05 — Build the tensor catalog](./05-build-tensor-catalog.md)
