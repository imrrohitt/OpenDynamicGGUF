# Step 07 — Build the calibration corpus (3-way split)

← [06 Weight features](./06-compute-weight-features.md) · [Index](./README.md) · Next: [08 Activation features](./08-compute-activation-features.md) →

---

## Goal

Create the **text** the model will be run on for activation stats, imatrix, KL search, and final validation — and split it so search never sees the judging set.

---

## Command

```bash
odg corpus --model functiongemma:latest
# larger production-sized corpus:
odg corpus --model functiongemma:latest --target-tokens 300000 --force
odg status --model functiongemma:latest
```

Requires Step 06 done. Default `--target-tokens 50000` (plumbing). Use `300000+` for production.

---

## Why it exists

Activation stats, imatrix, and ΔKLD all need **real prompts**. Wikipedia-only calibration overfits Wikipedia-style KL scores. Instruct models also need **chat-template** formatting.

---

## Inputs / Outputs

| | |
|---|---|
| **Input** | Resolve descriptor (`chat_template`, `specialty_domain`) |
| **Output** | `calib.txt`, `search.txt`, `heldout.txt`, `corpus_manifest.json` |

```text
steps/07_corpus/
  calib.txt             # ~60% — imatrix + activation features
  search.txt            # ~20% — ΔKLD during probe / optimize
  heldout.txt           # ~20% — validation ONLY
  corpus_manifest.json
  output.json
  status.json
  log.txt
```

Hard rule: **optimizer must not read held-out.**

---

## How it works

Mixed offline banks (no download): conversation / code / math / multilingual / specialty.

For FunctionGemma (`specialty_domain=function_calling`), includes tool-call traces rendered with the Gemma chat template:

```text
<start_of_turn>user
What's the weather in Paris?
<end_of_turn>
<start_of_turn>model
call weather_api(city="Paris")
<end_of_turn>
```

Then tiles with light variants until `--target-tokens`, and splits 60/20/20 (`seed=42`).

---

## Done when

- [x] Three files written with disjoint content
- [x] Chat template applied for instruct models
- [x] Domain data included when specialty_domain is set
- [x] Token counts logged in a corpus manifest

## Next

[Step 08 — Compute activation features](./08-compute-activation-features.md)
