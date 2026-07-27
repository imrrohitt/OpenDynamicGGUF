# Feature 03 — Interactive HTML report

← [02 Benchmark runner](./02-benchmark-runner.md) · [Index](./README.md) · Next: [04 Recipe marketplace](./04-recipe-marketplace.md) →

Priority: ⭐⭐⭐⭐⭐ · Phase 1 · New modules: `report.py` · New command: `odg report`

---

## Goal

```bash
odg report --model gemma4-27b
  → artifacts/runs/<run>/report.html
```

A single self-contained HTML file that makes the whole optimization transparent:

- tensor importance & sensitivity heatmap (role × depth)
- final layer/group allocations (which bits went where, and why)
- byte distribution (where the gigabytes actually live)
- KLD distribution (mean, p99.9, max — the same numbers the gates check)
- perplexity + benchmark comparison vs BF16 (from feature 02, when present)
- size ↔ quality Pareto frontier with the chosen point highlighted

---

## Why it exists

Trust. A closed dynamic quant says "believe us"; this project's pitch is *every decision
traces to a measured number*. The run store already contains all those numbers — this feature
just makes them visible without reading fifteen `output.json` files. It is also the artifact
users share (and feature 06 uploads to HF alongside the GGUF).

---

## Depends on

- Run artifacts that already exist: catalog (05), weight/activation features (06/08),
  sensitivity table (12), recipe + Pareto set (13), validation metrics (15).
- Feature 02's `benchresult.json` — optional section, rendered when present.

---

## Design

### Principles

1. **Read-only over the run store.** `report.py` never computes new metrics; it renders
   artifacts that steps already wrote. Missing artifact → section rendered as "not run",
   never an error.
2. **Self-contained file.** Inline JS/CSS (single vendored chart lib), data embedded as JSON.
   Must work offline, from `file://`, and attach to an HF repo or GitHub release as one file.
3. **Every number links to its source.** Each chart caption carries the artifact path + hash
   it came from (platform invariant 4).

### Sections → source artifacts

| Section | Source |
|---|---|
| Model summary | `01_resolve/output.json`, `05_catalog` |
| Byte distribution treemap | catalog shapes × final recipe types |
| Sensitivity heatmap (role × depth × quant) | `12_sensitivity` table |
| Allocation table with reasons | `13_optimize/recipe.yaml` + sensitivity rows (feature 10 deepens this) |
| KLD distribution | `15_validate` Tier-1 metrics |
| Pareto frontier | `13_optimize/pareto/*.yaml` |
| Benchmarks | `benchresult.json` (feature 02) |
| Reproducibility block | source SHA, imatrix SHA, recipe hash, command line |

---

## Build steps

1. **Data extractor.** One function per section: run dir → plain JSON dict. Pure, unit-tested
   against a fixture run (the existing `artifacts/runs/20260727-*` layout is the fixture).
2. **Skeleton template.** Static HTML shell + embedded `window.ODG_DATA = {...}`; sections
   render "not run" gracefully.
3. **Charts, cheapest first.** Allocation table → byte treemap → sensitivity heatmap → KLD
   histogram → Pareto scatter. One PR each.
4. **Benchmark section.** Render `benchresult.json` with paired-delta CIs shown as error
   bars, not bare scores.
5. **`odg report` CLI** + auto-generation at the end of `odg run` / `odg fit` when step 15
   completes.
6. **Source-hash captions.** Wire artifact path + sha into every section footer.

---

## Done when

- [ ] `odg report` on an existing run emits one self-contained `report.html`
- [ ] Works with a partial run (e.g. stopped at step 12) — sections degrade, never crash
- [ ] All six core visualizations render from real run data
- [ ] Every section captions its source artifact + hash
- [ ] File opens correctly offline from `file://`

## Next

[Feature 04 — Recipe marketplace](./04-recipe-marketplace.md)
