# Feature 07 — Automatic calibration dataset builder

← [06 Hugging Face integration](./06-huggingface-integration.md) · [Index](./README.md) · Next: [08 Recipe search (AutoML)](./08-recipe-search-automl.md) →

Priority: ⭐⭐⭐⭐☆ · Phase 3 · New modules: `calib_builder.py` (extends `corpus.py`) · CLI: `odg corpus --workload …`

---

## Goal

Answer "which calibration dataset should I use?" automatically, with open recipes for the
datasets themselves:

```bash
odg corpus --model qwen3 --workload coding
odg corpus --model gemma4 --workload tools --target-tokens 800000
odg corpus --model llama4 --mix "chat:0.4,coding:0.3,math:0.2,multilingual:0.1"
```

Workloads: `coding · chat · reasoning · math · tools · long-context · general` (+ plugin-provided).

---

## Why it exists

Calibration data quality is known to matter a lot, but good corpora are curated privately —
the community mostly falls back to wikitext, which doesn't even exercise chat templates. An
**open calibration pipeline** (sources + mixing weights + rendering, all versioned) makes
every imatrix and sensitivity table reproducible end-to-end, not just the quantization step.

Step 07 already implements the 60/20/20 three-way split and target domain shares. This
feature upgrades *where the text comes from* and makes workload mixes first-class.

---

## Depends on

- Step 07 (`corpus.py`) — this extends it; the split logic and hard walls are untouched.
- `datasets` (HF) for source loading.

---

## Design

### Corpus recipe (`odg/corpus/v1`)

The dataset itself gets a recipe, so a corpus id means something:

```yaml
schema: odg/corpus/v1
id: odg-coding-v1
target_tokens: 500000
seed: 42
sources:
  - dataset: bigcode/the-stack-smol       # pinned revision
    revision: "abc123"
    share: 0.5
    filters: {languages: [python, js, rust], max_len: 4096}
  - dataset: open-chat-traces
    revision: "def456"
    share: 0.3
    render: chat_template                  # rendered with the target model's template
  - dataset: gsm8k
    revision: "…"
    share: 0.2
render:
  chat_template: from_model               # tokenizer_config of the resolved model
splits: {calib: 0.6, search: 0.2, heldout: 0.2}
```

Content hash of (recipe + resolved revisions) = corpus id recorded in `recipe.yaml`
(`calibration.corpus_id` already exists in the recipe format).

### Workload presets are corpus recipes

`--workload coding` just selects a maintained `odg/corpus/v1` file from
`calib/` in this repo — reviewable, forkable, PR-able like marketplace recipes. `--mix`
builds an ad-hoc recipe from the preset sources.

### Chat-template rendering

Instruct models get every sample rendered through their own chat template (weakness of most
community imatrix data, already called out in the step 07 doc). Tool-calling workloads render
real schemas + calls so activations match the deployment distribution.

---

## Build steps

1. **Corpus recipe schema.** Parser/validator + content-hash id. `corpus.py` learns to run
   from a recipe instead of only its built-in defaults.
2. **Source loaders.** HF `datasets` with pinned revisions, per-source filters, token
   accounting toward shares.
3. **Chat-template renderer.** From the resolved model's tokenizer config; raw-text fallback
   for base models.
4. **Preset library.** `calib/general.yaml` first (mirrors current defaults), then `coding`,
   `chat`, `math`, `reasoning`, `tools`, `long-context`. Each is a small reviewed PR.
5. **CLI wiring.** `--workload`, `--mix`, `--corpus-recipe path.yaml` on `odg corpus`;
   `odg fit --workload X` passes it through.
6. **Provenance hookup.** Corpus id + recipe hash recorded into `recipe.yaml` and shown in
   report/model card.
7. **A/B evidence.** For one model, publish sensitivity/KLD deltas of `coding` vs `general`
   calibration — the doc that proves the feature matters.

---

## Done when

- [ ] `odg corpus --workload coding` builds a corpus from a pinned, versioned recipe
- [ ] Same recipe + same seed → identical split hashes (bit-for-bit corpus reproducibility)
- [ ] Instruct models are chat-template rendered; verified by inspecting emitted samples
- [ ] Corpus id flows into `recipe.yaml`, report, and model card
- [ ] At least 4 workload presets maintained in-tree

## Next

[Feature 08 — Recipe search (AutoML)](./08-recipe-search-automl.md)
