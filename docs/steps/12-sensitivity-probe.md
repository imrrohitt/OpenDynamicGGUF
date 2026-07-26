# Step 12 — Sensitivity probing (trial quantize + measure ΔKLD)

← [11 Logits](./11-cache-reference-logits.md) · [Index](./README.md) · Next: [13 Optimize](./13-optimize-recipe.md) →

---

## Goal

For each **tensor group**, try one or more quantization levels (only that group changed; rest stays high precision), measure **ΔKLD** and **Δbytes**, and fill the sensitivity table the optimizer uses.

This is the heart of OpenDynamicGGUF.

---

## Why it exists

Weight/activation features can look identical for two layers while ΔKLD after Q4 is `0.002` vs `0.12`. **Only a real probe measures sensitivity.**

Rule:

| Features (Steps 06–08) | Probe (this step) |
|---|---|
| Rank groups; choose which Q levels to try first | Accept / reject; produce Δbytes / ΔKLD |

---

## Inputs / Outputs

| | |
|---|---|
| **Input** | Catalog + features, BF16 GGUF, imatrix, `logits-search.bin` |
| **Output** | Sensitivity table: `(group, quant) → (Δbytes, ΔKLD, …)` |

---

## How it works

```text
For each group (role × depth), prioritized by feature score:
      │
      ▼
Pick trial types (e.g. Q3, Q4, Q5) — easy groups try lower first
      │
      ▼
llama-quantize --imatrix … --tensor-type "<group regex>=qX_k"
  (only that group; baseline elsewhere high / pure)
      │
      ▼
llama-perplexity --kl-divergence-base logits-search.bin --kl-divergence
      │
      ▼
Record ΔKLD, top-token agreement, Δbytes
      │
      ▼
Score = bytes_saved / max(ΔKLD, ε)
```

### Example probe command

Only mid-layer `ffn_up` → Q3_K:

```bash
./llama-quantize --imatrix imatrix.gguf \
  --tensor-type "\.(1[0-9])\.ffn_up=q3_k" \
  model-bf16.gguf probe-ffnup-mid-q3.gguf q6_k

./llama-perplexity -m probe-ffnup-mid-q3.gguf \
  --kl-divergence-base logits-search.bin --kl-divergence
```

### Illustrative sensitivity table

| Group | Probe | Size saved | ΔKLD | Decision hint |
|---|---|---|---|---|
| `ffn_up@middle` | Q4→Q3 | −310 MB | +0.004 | Good ratio → compress |
| `ffn_gate@middle` | Q4→Q3 | −305 MB | +0.005 | Good ratio → compress |
| `ffn_down@middle` | Q4→Q3 | −180 MB | +0.019 | Keep higher |
| `attn_v@all` | Q6→Q4 | −45 MB | +0.037 | Pin Q6 |
| `embedding` | Q8→Q4 | −190 MB | +0.055 | Pin Q8 |

---

## Example: identical stats, different KL

```text
Layer A & B: same mean/var/sparsity/norm
Probe Q4:
  Layer A ΔKLD = 0.002   → compress
  Layer B ΔKLD = 0.12    → do not compress
```

Features would have ranked them the same; the probe separates them.

---

## Done when

- [ ] All quantizable groups probed for at least one bit-width
- [ ] Table persisted (`sensitivity.json`) with hashes of inputs
- [ ] Search split only (not held-out)

## Next

[Step 13 — Optimize the recipe](./13-optimize-recipe.md)
