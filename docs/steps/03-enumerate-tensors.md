# Step 03 — Enumerate every tensor

← [02 Load](./02-load-model.md) · [Index](./README.md) · Next: [04 Classify](./04-classify-tensors.md) →

---

## Goal

List **every** parameter (and relevant buffer) in the model: name, shape, dtype, nbytes.

At the end of this step you know *what exists* — not yet what role it plays or how sensitive it is.

---

## Why it exists

You cannot assign bits to tensors you have not discovered. Different families name layers differently (`q_proj` vs `wq`); enumeration is the universal first scan.

---

## Inputs / Outputs

| | |
|---|---|
| **Input** | Loaded model (Step 02) |
| **Output** | Flat list: `{name, shape, dtype, nbytes}` |

---

## How it works

PyTorch exposes parameters via `state_dict()`:

```python
rows = []
for name, tensor in model.state_dict().items():
    rows.append({
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "nbytes": int(tensor.numel() * tensor.element_size()),
    })
```

Typical dense CausalLM names:

```text
model.embed_tokens.weight
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.k_proj.weight
model.layers.0.self_attn.v_proj.weight
model.layers.0.self_attn.o_proj.weight
model.layers.0.mlp.gate_proj.weight
model.layers.0.mlp.up_proj.weight
model.layers.0.mlp.down_proj.weight
model.layers.0.input_layernorm.weight
model.layers.0.post_attention_layernorm.weight
model.norm.weight
lm_head.weight
```

MoE models add expert / router names; hybrids add `ssm_*`. Record whatever appears.

---

## Example (tiny model)

```text
# gemma-3-270m style (illustrative counts)

model.embed_tokens.weight          shape=(262144, 640)   ~320 MB @ bf16
model.layers.0.self_attn.q_proj…   shape=(640, 640)      ~0.8 MB
…
lm_head.weight                     shape=(262144, 640)   (may tie with embed)

Total tensors: ~150
Total params:  ~270M
```

Print a quick inventory:

```python
print(f"tensors={len(rows)}  params={sum(r['nbytes'] for r in rows)/2:.0f} (bf16 elems approx)")
for r in rows[:5]:
    print(r["name"], r["shape"])
```

---

## Done when

- [ ] Every `*.weight` (and needed buffers) is listed
- [ ] Shapes and nbytes known
- [ ] No silent skip of MoE/SSM tensors if present

## Next

[Step 04 — Classify each tensor by role](./04-classify-tensors.md)
