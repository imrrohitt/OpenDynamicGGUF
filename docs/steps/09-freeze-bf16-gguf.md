# Step 09 — Freeze the GGUF reference

← [08 Activations](./08-compute-activation-features.md) · [Index](./README.md) · Next: [10 Imatrix](./10-build-imatrix.md) →

---

## Goal

Produce one **frozen, hashed GGUF** that all llama.cpp probes, imatrix, and exports use.

Ideal: HF BF16 → `model-bf16.gguf`.  
Pragmatic (current Ollama path): promote the working GGUF → `model-ref.gguf` + SHA.

---

## Command

```bash
odg freeze-gguf --model functiongemma:latest

# Require true BF16 (needs HF weights + llama.cpp converter):
export LLAMA_CPP_DIR=~/llama.cpp
odg resolve --model functiongemma:latest --prefer-hf --download-weights --force
odg freeze-gguf --model functiongemma:latest --mode hf-convert --require-bf16 --force
```

Requires Step 08 done.

---

## Modes

| Mode | Behavior |
|---|---|
| `auto` | HF convert if possible; else promote source GGUF |
| `hf-convert` | Fail unless `convert_hf_to_gguf.py` + HF dir work |
| `promote` | Always hardlink/symlink/copy the resolve/load GGUF |

---

## Outputs

```text
steps/09_freeze_gguf/
  model-bf16.gguf          # or model-ref.gguf if not BF16
  model-*.gguf.sha256
  freeze_manifest.json
  output.json
  status.json
  log.txt
```

Also verifies catalog tensor names ⊆ GGUF tensor index.

---

## Done when

- [x] Frozen GGUF exists under the step dir with SHA-256
- [x] Catalog names checked against the file
- [x] Manifest records whether it is a true BF16 reference

## Next

[Step 10 — Build the importance matrix](./10-build-imatrix.md)
