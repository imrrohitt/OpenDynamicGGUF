# Step 11 — Cache BF16 reference logits

← [10 Imatrix](./10-build-imatrix.md) · [Index](./README.md) · Next: [12 Probe](./12-sensitivity-probe.md) →

---

## Goal

Run the BF16 GGUF once on **search** and **held-out** text and save logit distributions. Every quantized candidate will be compared to these caches via KL divergence — without re-running BF16 each time.

---

## Why it exists

ΔKLD needs:

```text
KL( P_BF16  ‖  P_quant )
```

Computing `P_BF16` is expensive. Cache it.

---

## Inputs / Outputs

| | |
|---|---|
| **Input** | `model-bf16.gguf` + `search.txt` + `heldout.txt` |
| **Output** | `logits-search.bin`, `logits-heldout.bin` |

---

## How it works

```bash
# Search split — used during probing / optimization
./llama-perplexity \
  -m model-bf16.gguf \
  -f search.txt \
  --kl-divergence-base logits-search.bin

# Held-out — used ONLY in validation (Step 15)
./llama-perplexity \
  -m model-bf16.gguf \
  -f heldout.txt \
  --kl-divergence-base logits-heldout.bin
```

(Exact flags can vary by llama.cpp version — wrap in `runners.py`.)

Later, for a candidate:

```bash
./llama-perplexity \
  -m candidate.gguf \
  --kl-divergence-base logits-search.bin \
  --kl-divergence
```

---

## Example

```text
search.txt     → logits-search.bin     (optimizer objective)
heldout.txt    → logits-heldout.bin    (Tier-1 gate only)

Cache key includes: bf16_sha + corpus_sha + llama.cpp version
```

---

## Done when

- [ ] Both logit caches exist
- [ ] Search cache never mixed with held-out
- [ ] Cached under content-addressed store

## Next

[Step 12 — Sensitivity probing](./12-sensitivity-probe.md)
