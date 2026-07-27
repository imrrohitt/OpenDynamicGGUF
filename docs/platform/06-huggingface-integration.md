# Feature 06 — Hugging Face integration

← [05 Public leaderboard](./05-public-leaderboard.md) · [Index](./README.md) · Next: [07 Calibration dataset builder](./07-calibration-dataset-builder.md) →

Priority: ⭐⭐⭐⭐⭐ · Phase 2 · New modules: `publish.py` · New command: `odg publish`

---

## Goal

One command from a Hub model id to a published, documented, benchmarked GGUF repo:

```bash
odg publish --model unsloth/Qwen3.6 --gpu 16GB --hf-repo me/Qwen3.6-UD
```

which automatically:

1. downloads the BF16 source (resolver already handles HF refs),
2. optimizes (full pipeline, hardware-aware budget from feature 01),
3. benchmarks (feature 02),
4. generates the model card (feature 11; a minimal built-in card until then),
5. uploads GGUF(s) + `recipe.yaml` + `report.html` + `benchresult.json`.

---

## Why it exists

Publishing quantized models today is a dozen manual steps, so most quants ship with no
recipe, no report, and a copy-pasted model card. Making the *documented, reproducible*
release the path of least resistance is how the recipe/report/benchresult formats spread
through the ecosystem.

---

## Depends on

- Features 01 (budget), 02 (benchmarks), 03 (report). Feature 11 upgrades the card later.
- `huggingface_hub` for auth + uploads (resolver already depends on it for downloads).

---

## Design

### Repo layout produced

```text
me/Qwen3.6-UD/
  README.md                     # model card (auto-generated)
  qwen3.6-UD-Q4~10.8GB.gguf     # one file per published Pareto point
  recipes/
    rtx-16gb.yaml               # odg/recipe/v2, one per GGUF
  reports/
    report.html
    benchresult.json
```

Multiple Pareto points can be published to one repo (`--pareto top3`); each GGUF pairs with
its own recipe.

### Behavior rules

- **Resumable like everything else.** `odg publish` is steps on top of the run store; a
  failed upload resumes without re-optimizing. Uploads are the last step and idempotent
  (hash-checked before re-upload).
- **No unmeasured upload.** Refuses to upload if validation gates (step 15) failed;
  `--allow-ungated` exists but stamps the model card with a visible warning.
- **Provenance embedded twice.** In GGUF KV metadata (`--override-kv`, already designed in
  step 14) and in the model card's reproduce block:

```bash
# Reproduce this file bit-for-bit
odg recipe build recipes/rtx-16gb.yaml   # verifies source + imatrix hashes
```

---

## Build steps

1. **Upload primitives.** `publish.py`: create repo, upload file with hash-skip, retry.
   Tested against a scratch HF repo.
2. **Minimal model card.** Template with model summary, size table, KLD/gate numbers,
   reproduce block. (Feature 11 replaces this with the full generator.)
3. **`odg publish` for an existing finished run.** No new computation — collect artifacts,
   render card, upload. This alone is already valuable.
4. **End-to-end mode.** Missing steps are executed first (delegate to `odg fit` driver),
   then publish. `--pareto topN` publishes multiple points.
5. **Gate enforcement.** Wire the refuse/`--allow-ungated` logic + card warning stamp.
6. **Benchmark integration.** Run/attach `benchresult.json`; card table gets paired deltas.
7. **Leaderboard hand-off.** Print the ready-to-run `odg leaderboard submit` command after a
   successful publish.

---

## Done when

- [ ] One command takes a HF model id to a complete uploaded repo (GGUF + recipe + report + card)
- [ ] Interrupted publish resumes without recomputation or duplicate uploads
- [ ] Gated: failed validation blocks upload by default
- [ ] Published GGUF carries provenance KV metadata; card carries the reproduce block
- [ ] Round-trip verified: `odg recipe build` on the uploaded recipe reproduces the uploaded
      GGUF hash

## Next

[Feature 07 — Calibration dataset builder](./07-calibration-dataset-builder.md)
