# Feature 11 — Automatic model cards

← [10 Explainability](./10-explainability.md) · [Index](./README.md) · Next: [12 Plugin system](./12-plugin-system.md) →

Priority: ⭐⭐⭐⭐☆ · Phase 4 · New modules: `modelcard.py` · New command: `odg card`

---

## Goal

```bash
odg card --model gemma4-27b
  → artifacts/runs/<run>/README.md     # complete HF-ready model card
```

Generated from run artifacts, covering everything a maintainer writes by hand today:

- optimization settings (budget, objective, pins) and the recipe summary
- calibration dataset (corpus recipe id + composition — feature 07)
- benchmark results as paired deltas vs BF16 with CIs (feature 02)
- supported hardware (which profiles this file fits, at which context — feature 01)
- memory usage table (weights + KV cache at 4k/8k/32k ctx)
- throughput (measured per device where available, predicted labeled as such)
- known limitations (failed/waived gates, workload caveats from the corpus mix)
- reproduce block (`odg recipe build …` + all hashes)

---

## Why it exists

Model cards for quants are either missing or copy-pasted boilerplate, because writing an
honest one requires collecting exactly the numbers this pipeline already produces. Generating
the card makes documentation a free by-product — and makes `odg publish` (feature 06) ship
complete repos by default.

---

## Depends on

- Data producers: features 01, 02, 03, 07, 10 (each contributes a section; sections degrade
  gracefully when the producer hasn't run).
- Consumer: feature 06 (`odg publish` uses this instead of its minimal built-in card).

---

## Design

- **Template + data, strictly separated.** A Jinja template renders a `carddata.json`
  assembled from run artifacts. Users can supply `--template mine.md.j2`; the data contract
  is the stable API.
- **HF metadata block included.** YAML front-matter (`base_model`, `license` inherited from
  the source repo, `tags: [gguf, odg, dynamic-quant]`, `quantized_by`) so cards index
  correctly on the Hub.
- **Multi-file aware.** One card can describe several published Pareto points; the size /
  quality / speed table has one row per GGUF, generated from each file's recipe + results.
- **Honesty rules carry over.** Predicted vs measured labeled; missing sections say "not
  measured", never disappear silently when data *should* exist (a gate summary is always
  present, even if it says "gates waived with --allow-ungated ⚠").

---

## Build steps

1. **`carddata.json` assembler.** One function per section over the run store (mirrors the
   report extractor from feature 03 — share code where sections overlap).
2. **Default template.** Match the structure/quality of the best hand-written community
   cards; include the front-matter block.
3. **Memory table generator.** Weights + KV math (feature 01's calculator) at standard
   context lengths.
4. **Known-limitations section.** Derived from gate outcomes, corpus mix (e.g. "calibrated
   for coding; chat quality not specifically validated"), and waivers.
5. **`odg card` CLI** + `--template` override.
6. **Publish integration.** Feature 06 swaps in this generator; minimal card retired.

---

## Done when

- [ ] Card generated from any finished run with zero manual input
- [ ] Front-matter validates on the HF Hub (renders with correct base-model link + tags)
- [ ] All numbers trace to artifacts; predicted vs measured labeled
- [ ] Multi-GGUF repos get one coherent card with a per-file table
- [ ] A maintainer can restyle via `--template` without touching Python

## Next

[Feature 12 — Plugin system](./12-plugin-system.md)
