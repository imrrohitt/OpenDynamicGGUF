# Step 08 — Compute activation features

← [07 Corpus](./07-build-calibration-corpus.md) · [Index](./README.md) · Next: [09 Freeze GGUF](./09-freeze-bf16-gguf.md) →

---

## Goal

Run **calibration text** through the BF16 model and record **activation** statistics per layer/tensor group (range, outliers, which channels fire hard).

---

## Why it exists

Weights alone cannot tell you which neurons matter on real workloads.

```text
Prompt A "What is AI?"     → layer12 channel k = 0.2
Prompt B "Write Python…"   → layer12 channel k = 15.2   ← spike!
```

That spike may deserve higher precision. Activation features capture this for **ranking**.

(Full importance for llama.cpp rounding is Step 10 / `llama-imatrix`.)

---

## Inputs / Outputs

| | |
|---|---|
| **Input** | BF16 model + `calib.txt` + catalog |
| **Output** | `activation_features` in catalog |

---

## How it works

```text
calib prompts
      │
      ▼
Forward pass (BF16)
      │
      ▼
Hook hidden states / linear inputs
      │
      ▼
Per tensor / group:
  activation range [min, max]
  activation outlier ratio
  optional per-channel max / RMS
```

Sketch with hooks:

```python
stats = {}  # name → running min/max/absmax

def make_hook(name):
    def hook(_m, inputs, _out):
        x = inputs[0].detach().float()
        s = stats.setdefault(name, {"min": 1e9, "max": -1e9, "absmax": 0.0})
        s["min"] = min(s["min"], x.min().item())
        s["max"] = max(s["max"], x.max().item())
        s["absmax"] = max(s["absmax"], x.abs().max().item())
    return hook

for name, mod in model.named_modules():
    if isinstance(mod, torch.nn.Linear):
        mod.register_forward_hook(make_hook(name))

for batch in calib_batches:
    model(**batch)
```

Merge into catalog:

```json
"activation_features": {
  "range_min": -4.2,
  "range_max": 15.2,
  "outlier_ratio": 0.003
}
```

---

## Example story

```text
Same weight stats for Layer A and Layer B.
Activation pass shows Layer B has huge absmax on coding prompts.
→ Rank Layer B as harder; try higher bits first in Step 12.
→ Still VERIFY with ΔKLD probe — activations only prioritize.
```

---

## Done when

- [ ] Calib forward pass completed (or sampled enough tokens)
- [ ] Activation features written for quantizable groups
- [ ] Catalog updated / re-hashed

## Next

[Step 09 — Freeze BF16 GGUF](./09-freeze-bf16-gguf.md)
