# Step 11 — Cache reference logits

← [10 Imatrix](./10-build-imatrix.md) · [Index](./README.md) · Next: [12 Probe](./12-sensitivity-probe.md) →

---

## Goal

Run the reference GGUF once on **search** and **held-out** text and save logit distributions. Every quantized candidate is compared to these caches via KL divergence — without re-running the reference each time.

---

## Command

```bash
odg reference-logits --model functiongemma:latest

# require real caches:
export LLAMA_CPP_DIR=~/llama.cpp
odg reference-logits --model functiongemma:latest --mode llama --force
```

Requires Steps 07 + 09 + 10. Uses `search.txt` + `heldout.txt` only — **never calib**.

---

## Outputs

```text
steps/11_reference_logits/
  logits-search.bin           # when llama-perplexity succeeds
  logits-heldout.bin
  logits-*.bin.MISSING        # markers in proxy mode
  logits_manifest.json        # cache_key = hash(gguf, search, heldout)
  output.json
  status.json
  log.txt
```

Later probe:

```bash
llama-perplexity -m candidate.gguf \
  --kl-divergence-base logits-search.bin --kl-divergence
```

---

## Done when

- [x] Manifest + cache_key recorded
- [x] Search / heldout kept separate
- [x] Real `.bin` caches **or** documented MISSING markers

## Next

[Step 12 — Sensitivity probing](./12-sensitivity-probe.md)
