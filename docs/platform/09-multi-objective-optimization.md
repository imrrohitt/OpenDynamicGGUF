# Feature 09 — Multi-objective optimization

← [08 Recipe search (AutoML)](./08-recipe-search-automl.md) · [Index](./README.md) · Next: [10 Explainability](./10-explainability.md) →

Priority: ⭐⭐⭐⭐⭐ · Phase 3 · New modules: `objectives.py` · CLI: `odg fit --objective …`

---

## Goal

Optimize for what the user actually cares about, not only quality-per-byte:

```bash
odg fit --model qwen3 --gpu 12GB --objective quality        # default: today's behavior
odg fit --model qwen3 --gpu 12GB --objective speed          # tok/s-weighted
odg fit --model qwen3 --gpu 12GB --objective balanced
odg fit --model qwen3 --ram 16GB --objective "quality:0.5,speed:0.3,ram:0.2"
```

Objectives: `quality · speed · vram · ram · latency` (energy later, once measurable reliably).

---

## Why it exists

Different users sit at different points: a chatbot host wants latency, a laptop user wants
RAM headroom, a batch-eval user wants raw quality. Today the optimizer has exactly one
currency (bytes saved / ΔKLD). Multi-objective support turns the same sensitivity data into
different frontiers — and turns "pick a quant" into "state your priorities".

---

## Depends on

- Feature 01 (hardware profile supplies the device constants), feature 02 (throughput
  measurement validates predictions), feature 08 (frontier machinery — this feature mostly
  adds *scoring dimensions* to it).

---

## Design

### Where each objective's number comes from

| Objective | Source | Cost |
|---|---|---|
| quality | measured ΔKLD (existing) | probe/measure |
| vram / ram | exact bytes + KV cache math (feature 01) | free |
| speed (throughput) | model: bytes ÷ bandwidth per token, corrected by quant-type dequant cost table; validated by `llama-bench` spot checks | free (model), cheap (validation) |
| latency (first token) | prompt-processing model, same inputs | free (model) |

Speed/latency use a **predictive model calibrated per backend**, because measuring every
candidate is unaffordable. The model's error bars are reported, and final frontier candidates
get a real `llama-bench` measurement before anything is published (no unmeasured claim).

Not all quant types decode at equal speed at equal size (e.g. IQ formats trade dequant cost
for quality) — the dequant cost table is what makes this objective non-trivial and valuable.

### Scoring (`objectives.py`)

- Named presets = weight vectors over normalized objectives.
- Hard constraints stay hard: the hardware budget is never traded away by a weight.
- Output is still a frontier — now in more dimensions — with the preset selecting the
  default point; `report.html` shows the trade-off surface.

---

## Build steps

1. **Objective interface.** `score(candidate, context) → float` + `is_exact` flag; quality
   and vram/ram objectives wrap existing numbers. This is also the `odg.metrics` plugin ABC.
2. **Throughput model v0.** bytes ÷ bandwidth with per-quant-type dequant multipliers seeded
   from published llama.cpp benchmarks.
3. **Calibration harness.** Run `llama-bench` on 3–5 existing quants of one model, fit the
   multipliers for the local backend, store per hardware-profile id.
4. **Weighted frontier selection.** Presets + custom weight strings; wire into feature 08's
   selection (or step 13's Pareto-point choice before 08 lands).
5. **Latency objective.** Prompt-processing variant of the throughput model.
6. **Report integration.** Multi-axis frontier view; predicted vs measured speed shown with
   error bars.
7. **Validation gate.** Published frontier points require a real throughput measurement
   within the model's stated error; otherwise the number is labeled "predicted".

---

## Done when

- [ ] `--objective speed` produces a measurably faster (and slightly larger-KLD) pick than
      `--objective quality` on the same budget — demonstrated on one model
- [ ] Throughput predictions within ~15% of `llama-bench` on the calibrated backend
- [ ] Hard budget constraints never violated regardless of weights
- [ ] Predicted vs measured numbers clearly distinguished everywhere they surface
- [ ] Objectives implement the same interface plugins will use (feature 12)

## Next

[Feature 10 — Explainability](./10-explainability.md)
