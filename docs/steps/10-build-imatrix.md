# Step 10 — Build the importance matrix (imatrix)

← [09 Freeze GGUF](./09-freeze-bf16-gguf.md) · [Index](./README.md) · Next: [11 Logits](./11-cache-reference-logits.md) →

---

## Goal

Build activation-based importance stats from the **calib** split so quantization can protect important weights when rounding.

Ideal: `llama-imatrix` → `imatrix.gguf` for `llama-quantize --imatrix`.  
Fallback: `imatrix_proxy.json` from weight/activation features (plumbing only).

---

## Command

```bash
odg imatrix --model functiongemma:latest

# require real binary:
export LLAMA_CPP_DIR=~/llama.cpp   # must contain build/bin/llama-imatrix
odg imatrix --model functiongemma:latest --mode llama --force

# proxy only:
odg imatrix --model functiongemma:latest --mode proxy --force
```

Requires Steps 07 + 09. **Never** use heldout.txt here.

---

## Outputs

```text
steps/10_imatrix/
  imatrix.gguf              # when llama-imatrix succeeds
  imatrix_proxy.json        # readable per-tensor scores (always in proxy/auto-fallback)
  imatrix.gguf.MISSING      # marker when proxy-only
  imatrix_manifest.json
  output.json
  status.json
  log.txt
```

---

## Done when

- [x] Calib-only input recorded (same GGUF SHA as Step 09)
- [x] `imatrix.gguf` **or** documented `imatrix_proxy.json` produced
- [x] Hashes / paths stored in manifest

## Next

[Step 11 — Cache BF16 reference logits](./11-cache-reference-logits.md)
