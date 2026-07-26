# OpenDynamicGGUF — Step-by-step architecture

This folder breaks the full pipeline into **small, teachable steps**. Read them in order the first time; later use them as a reference while implementing.

Each step file answers the same questions:

| Section | Meaning |
|---|---|
| Goal | What this step achieves |
| Why it exists | Failure mode it prevents |
| Inputs / Outputs | What goes in and what comes out |
| How it works | Detailed explanation |
| Example | Concrete walkthrough |
| Done when | Checklist before moving on |
| Next | Link to the following step |

---

## Big picture

```text
01 Resolve model ref ──────────────► original BF16 HF checkpoint
02 Load model
03 Enumerate tensors (state_dict)
04 Classify roles (attn / mlp / …)
05 Build tensor catalog
06 Compute weight features ────────► (no text needed)
07 Build calibration corpus ───────► calib / search / held-out
08 Compute activation features ────► (needs calib text)
09 Freeze BF16 GGUF
10 Build imatrix
11 Cache reference logits
12 Sensitivity probe ──────────────► ΔKLD / Δbytes table
13 Optimize recipe ────────────────► bit assignment under budget
14 Export GGUF
15 Validate & release
```

---

## Step index

| # | Step | File | One-line summary |
|---|---|---|---|
| 00 | Checkpoint store | [00-checkpoint-store.md](./00-checkpoint-store.md) | Filesystem run store so no step is lost |
| 01 | Resolve model | [01-resolve-model.md](./01-resolve-model.md) · [impl](./01-resolve-model-impl.md) | Any user ref → original BF16 HF |
| 02 | Load model | [02-load-model.md](./02-load-model.md) · [impl](./02-load-model-impl.md) | Open GGUF/HF and build tensor index |
| 03 | Enumerate tensors | [03-enumerate-tensors.md](./03-enumerate-tensors.md) | Flat inventory: name / shape / dtype / nbytes |
| 04 | Classify tensors | [04-classify-tensors.md](./04-classify-tensors.md) | Map each name → role / depth / quantizable |
| 05 | Build catalog | [05-build-tensor-catalog.md](./05-build-tensor-catalog.md) | `tensor_catalog.json` source of truth |
| 06 | Weight features | [06-compute-weight-features.md](./06-compute-weight-features.md) | `odg weight-features` — mean/var/norms + hardness |
| 07 | Calibration corpus | [07-build-calibration-corpus.md](./07-build-calibration-corpus.md) | `odg corpus` — calib/search/heldout 60/20/20 |
| 08 | Activation features | [08-compute-activation-features.md](./08-compute-activation-features.md) | `odg activation-features` — hooks or proxy |
| 09 | Freeze BF16 GGUF | [09-freeze-bf16-gguf.md](./09-freeze-bf16-gguf.md) | `odg freeze-gguf` — hashed ref (BF16 or promote) |
| 10 | Build imatrix | [10-build-imatrix.md](./10-build-imatrix.md) | `odg imatrix` — llama-imatrix or proxy |
| 11 | Reference logits | [11-cache-reference-logits.md](./11-cache-reference-logits.md) | `odg reference-logits` — KL base caches |
| 12 | Sensitivity probe | [12-sensitivity-probe.md](./12-sensitivity-probe.md) | `odg sensitivity` — Δbytes/ΔKLD table |
| 13 | Optimize recipe | [13-optimize-recipe.md](./13-optimize-recipe.md) | `odg optimize` — greedy knapsack → recipe.yaml |
| 14 | Export GGUF | [14-export-gguf.md](./14-export-gguf.md) | Recipe → final quantized GGUF |
| 15 | Validate & release | [15-validate-and-release.md](./15-validate-and-release.md) | Tiered gates; ship or feedback |

---

## Related

- High-level overview: [`../../README.md`](../../README.md)
- Interactive canvas (if present): architecture canvas in Cursor project `canvases/`

**Rule to remember for every step:** statistics and features *prioritize*; measured ΔKLD *decides*; held-out validation *ships*.
