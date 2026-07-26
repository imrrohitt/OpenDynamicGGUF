# Step 12 — Sensitivity probing (trial quantize + measure ΔKLD)

← [11 Logits](./11-cache-reference-logits.md) · [Index](./README.md) · Next: [13 Optimize](./13-optimize-recipe.md) →

---

## Goal

For each **tensor group**, try quantization levels, record **ΔKLD** and **Δbytes**, and fill the sensitivity table the optimizer uses.

---

## Command

```bash
odg sensitivity --model functiongemma:latest
```

Requires Step 11. Uses **search** split only (never heldout).

Modes:
- `auto` / `proxy` — estimate ΔKLD from features + imatrix proxy; Δbytes from type sizes (plumbing)
- `llama` — reserved for real `llama-quantize` + `llama-perplexity --kl-divergence` (needs tools + logit caches)

---

## Outputs

```text
steps/12_sensitivity/
  sensitivity.json     # full (group, probe) → metrics table
  output.json          # summary + top efficiency / pin hints
  status.json
  log.txt
```

Each row includes: `delta_bytes`, `delta_kld`, `efficiency` (= bytes/ΔKLD), `tensor_type_regex`, `decision_hint`.

---

## Done when

- [x] All quantizable groups have ≥1 probe row
- [x] Table persisted with input hashes
- [x] Search-only (documented)

## Next

[Step 13 — Optimize the recipe](./13-optimize-recipe.md)
