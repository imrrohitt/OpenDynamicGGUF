# Step 10 — Build the importance matrix (imatrix)

← [09 Freeze GGUF](./09-freeze-bf16-gguf.md) · [Index](./README.md) · Next: [11 Logits](./11-cache-reference-logits.md) →

---

## Goal

Run `llama-imatrix` on the **calib** split to produce activation-based importance statistics that guide how `llama-quantize` rounds weights inside each tensor.

---

## Why it exists

Uniform quantization treats every weight equally. Imatrix says: “these weights matter more under real activations — protect them when rounding.”

```text
calib.txt → forward pass on BF16 GGUF → collect / average activations → imatrix.gguf
```

This is complementary to Step 08 (PyTorch activation features for *ranking*). Imatrix is what llama.cpp *consumes* during quantize.

---

## Inputs / Outputs

| | |
|---|---|
| **Input** | `model-bf16.gguf` + `calib.txt` |
| **Output** | `imatrix.gguf` (or `.dat` depending on build) |

---

## How it works

```bash
./llama-imatrix \
  -m model-bf16.gguf \
  -f calib.txt \
  -o imatrix.gguf \
  --chunks 100          # example flag; check your llama.cpp version
```

What happens inside (conceptually):

1. Load BF16 GGUF  
2. Feed calibration chunks  
3. Record activation magnitudes related to each weight tensor  
4. Average / store importance  
5. Write file for `--imatrix` in `llama-quantize`

---

## Example

```text
calib.txt:   ~900k tokens, chat+code+math, template-rendered
runtime:     tens of minutes on one GPU / Apple Silicon
output:      imatrix.gguf (~few MB–tens of MB)
```

**Do not** build imatrix on held-out text. Calib only.

---

## Done when

- [ ] `imatrix.gguf` produced from **calib** split only
- [ ] File hash stored for recipe provenance
- [ ] Same BF16 GGUF SHA as Step 09

## Next

[Step 11 — Cache BF16 reference logits](./11-cache-reference-logits.md)
