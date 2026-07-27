# Feature 05 — Public leaderboard

← [04 Recipe marketplace](./04-recipe-marketplace.md) · [Index](./README.md) · Next: [06 Hugging Face integration](./06-huggingface-integration.md) →

Priority: ⭐⭐⭐⭐⭐ · Phase 2 · New tree: `leaderboard/` · Data source: `benchresult.json` + recipe registry

---

## Goal

A public site where every optimized model is comparable at a glance:

```text
OpenDynamicGGUF Leaderboard
  Gemma · Qwen · Llama · DeepSeek · Mistral · GLM · Phi

  | Recipe        | Size    | PPL  | mean KLD | MMLU Δ vs BF16    | tok/s (device) |
  |---------------|---------|------|----------|--------------------|----------------|
  | rtx-3060.yaml | 10.8 GB | 6.91 | 0.0074   | −0.2 [−0.9, +0.5]  | 34.2 (3060)    |
```

Users compare, pick, click through to the recipe and full report — and naturally compete to
contribute better rows.

---

## Why it exists

Comparison drives contribution. Once recipes are shareable (feature 04) and results are
standardized (feature 02), a leaderboard is the flywheel: it makes quality visible, rewards
recipe authors with attribution, and gives newcomers an obvious "which quant should I
download" answer that no HF model-page scatter provides today.

---

## Depends on

- Feature 02 (`odg/benchresult/v1` is the only accepted evidence format).
- Feature 04 (rows are recipes; the registry is the submission channel).
- Feature 03 (each row links to its `report.html`).

---

## Design

### Static site, data-in-git

No backend. The leaderboard is a static site generated from checked-in JSON:

```text
leaderboard/
  data/<family>/<recipe-id>/benchresult.json   # submitted via PR, CI-verified
  generate.py                                   # data → static HTML/JSON
  site/                                          # generated output (GitHub Pages)
```

Same trust model as the marketplace: submission is a PR, CI validates, git history is the
audit log.

### Row integrity rules

A row is accepted only if:

1. `benchresult.json` is schema-valid and self-consistent,
2. it references a recipe that exists in the registry (`recipe_sha256` matches),
3. quality numbers carry paired deltas + CIs vs the declared BF16 reference,
4. the GGUF hash matches what the recipe reproduces (`odg scan`-style check, see feature 13).

Throughput numbers are tagged with the hardware profile id and shown per-device — never
averaged across devices.

### Views

- Per-family table (size / PPL / KLD / benchmark deltas / tok-s), sortable.
- Size-vs-quality scatter with the Pareto frontier drawn — the "which point do I want" view.
- Per-row detail: recipe YAML, full benchresult, link to `report.html`, author credit.

---

## Build steps

1. **Row validator.** The four integrity rules as a library function — shared by CI and by
   `odg leaderboard submit` (local pre-check).
2. **Generator MVP.** `generate.py`: data dir → one static page per family with the sortable
   table. No styling polish yet.
3. **Seed data.** Publish our own runs (the same ones seeding the marketplace) as the first
   rows.
4. **Scatter + frontier view.** Reuse the Pareto plotting from feature 03's report.
5. **CI + GitHub Pages.** PR check runs the validator; merge to main regenerates and deploys.
6. **Submission ergonomics.** `odg leaderboard submit` packages benchresult + recipe ref and
   prints the PR instructions (or opens it via `gh`).
7. **Cross-links.** Marketplace `list` shows leaderboard rank; leaderboard rows link back to
   `odg recipe` one-liners.

---

## Done when

- [ ] Static site deploys from git data with ≥1 family populated
- [ ] Every row passes the four integrity rules in CI — unverifiable rows are impossible
- [ ] Quality shown as paired delta + CI vs BF16, throughput shown per-device
- [ ] Each row links to recipe, benchresult, report, and author
- [ ] A stranger can submit a row with only `odg benchmark` + `odg leaderboard submit` + a PR

## Next

[Feature 06 — Hugging Face integration](./06-huggingface-integration.md)
