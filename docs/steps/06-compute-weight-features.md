# Step 06 — Compute weight features

← [05 Catalog](./05-build-tensor-catalog.md) · [Index](./README.md) · Next: [07 Calibration corpus](./07-build-calibration-corpus.md) →

---

## Goal

From each **weight tensor alone**, compute statistics used later for **ranking** (not final bit decisions).

No calibration text is required in this step.

---

## Command

```bash
odg weight-features --model functiongemma:latest
odg status --model functiongemma:latest
```

Requires Step 05 done. Optional: `--only-quantizable` to skip norms.

---

## Why it exists

Weight features are cheap and always available. They answer: “Does this matrix *look* easy or hard to quantize?” They do **not** answer: “What happens to the model if we actually quantize it?” That needs Steps 08 and 12.

---

## Inputs / Outputs

| | |
|---|---|
| **Input** | `tensor_catalog.json` + GGUF (F32 / F16 / BF16 / Q8_0) |
| **Output** | Updated catalog with `weight_features` + group aggregates |

```text
steps/06_weight_features/
  output.json           # summary + hardest/easiest groups
  tensor_catalog.json   # catalog with weight_features filled
  group_features.json   # group-level aggregates + hardness
  status.json
  log.txt
```

---

## How it works

```text
Weight tensor W  (dequant Q8_0 → f32 if needed)
      │
      ▼
mean, variance, entropy
sparsity (% |w| < 1e-3)
outlier_ratio (% beyond 6σ)
weight_norm (L2)
spectral_norm (power iteration, 2D mats ≤ 4096 dim)
      │
      ▼
group aggregates → hardness score (probe ranking)
```

**If the source is already Q8_0** (Ollama default): features come from dequantized Q8 weights. Fine for plumbing and relative ranking; for production quality ranking, re-run later on BF16/HF.

---

## Done when

- [x] Every readable tensor has `weight_features` (or skipped with reason)
- [x] Group-level aggregates + hardness ranking stored
- [x] Catalog hash updated

## Next

[Step 07 — Build the calibration corpus](./07-build-calibration-corpus.md)
