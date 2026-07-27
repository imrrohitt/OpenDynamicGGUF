# Feature 02 — Benchmark runner

← [01 Hardware-aware optimizer](./01-hardware-aware-optimizer.md) · [Index](./README.md) · Next: [03 HTML report](./03-report-visualization.md) →

Priority: ⭐⭐⭐⭐⭐ · Phase 1 · New modules: `benchmark.py` · New command: `odg benchmark`

---

## Goal

One command turns a GGUF into comparable, trustworthy numbers:

```bash
odg benchmark gemma-UD.gguf                       # standard suite
odg benchmark gemma-UD.gguf --suite coding        # HumanEval + LiveCodeBench focus
odg benchmark gemma-UD.gguf --tasks mmlu,gsm8k
```

producing `benchresult.json` (machine-readable) and feeding `benchmarks.html` (feature 03).

---

## Why it exists

Today everyone benchmarks quantized models manually, with different harness versions, few-shot
settings, and prompts — so numbers across the community are not comparable. A standard runner
with pinned configs makes every OpenDynamicGGUF model immediately comparable, and it is the
data source for the report (03), the leaderboard (05), and model cards (11).

This is also the productization of the existing **Tier-3 gate** in step 15: same statistical
methodology, promoted from an internal gate to a user-facing command.

---

## Depends on

- lm-eval-harness (llama.cpp / GGUF backend) — external.
- Step 15's paired-comparison rule (design principle 7): all deltas are **paired per-question
  vs the BF16 reference with confidence intervals**, never raw score thresholds.
- Optional: feature 01's hardware profile, to record throughput per device.

---

## Design

### Suites

| Suite | Tasks | Cost |
|---|---|---|
| `smoke` | tiny MMLU slice + 20 generations | minutes |
| `standard` | MMLU, GSM8K, HumanEval, TruthfulQA | hours |
| `coding` | HumanEval, MBPP, LiveCodeBench subset | hours |
| `long-context` | needle-in-haystack, RULER subset | hours |
| `tools` | schema-valid JSON rate on tool-call traces | ~1 hour |

Suites are data (a YAML of task ids + pinned few-shot/config), not code — so the plugin
system (feature 12) can later add suites via `odg.evals` entry points.

### Result schema (`odg/benchresult/v1`)

```json
{
  "schema": "odg/benchresult/v1",
  "gguf_sha256": "…",
  "recipe_sha256": "…",
  "reference": {"model": "bf16", "gguf_sha256": "…"},
  "harness": {"name": "lm-eval", "version": "0.4.x", "commit": "…"},
  "tasks": {
    "mmlu": {
      "score": 0.712,
      "paired_delta_vs_bf16": -0.002,
      "ci95": [-0.009, 0.005],
      "n": 14042
    }
  },
  "throughput": {"device": "rtx-3060-12gb", "pp_tps": 812.0, "tg_tps": 34.2},
  "memory": {"weights_gb": 10.8, "peak_vram_gb": 12.6, "ctx": 8192}
}
```

Every field the leaderboard or a model card will ever show comes from this file.

### Two measurement halves

1. **Quality** — lm-eval-harness over the GGUF via llama.cpp server/bindings, plus the BF16
   reference (cached per model: run once, reuse for every candidate — same trick as step 11's
   logit cache).
2. **Performance** — `llama-bench` style prompt-processing / token-generation throughput and
   peak memory on the current machine, tagged with the hardware profile id.

---

## Build steps

1. **Result schema + writer.** `odg/benchresult/v1` dataclass, validation, content-addressed
   storage in the run store. No runner yet.
2. **Throughput half.** Wrap `llama-bench` (discovery via existing `llama_bins.py`), parse
   output into the schema. Fast to build, immediately useful.
3. **Harness adapter.** One task (MMLU slice) end-to-end through lm-eval-harness against a
   GGUF. Pin harness version; record it in the result.
4. **BF16 reference cache.** Run the reference once per (model, suite), key by hash, reuse —
   candidates only pay their own eval cost.
5. **Paired statistics.** Per-question pairing + bootstrap CI, shared with step 15's Tier-3
   gate (extract the existing gate math into `benchmark.py` and have `validate.py` call it).
6. **Suite definitions.** `smoke` first (used in CI for this repo), then `standard`, then the
   domain suites.
7. **`odg benchmark` CLI.** `--suite`, `--tasks`, `--reference` (auto from run store when the
   GGUF came from a run), `--device` tag.
8. **Wire into step 15.** Tier-3 gate becomes "run `smoke`/`standard` suite and apply gate
   thresholds to the paired deltas".

---

## Done when

- [ ] `odg benchmark x.gguf --suite smoke` produces a valid `benchresult.json` in minutes
- [ ] Deltas are paired vs a cached BF16 reference with CIs — no raw-score-only output
- [ ] Harness version + task configs are pinned and recorded in the result
- [ ] Throughput + peak memory captured and tagged with a hardware profile id
- [ ] Step 15 Tier-3 consumes the same code path (one implementation of the statistics)

## Next

[Feature 03 — Interactive HTML report](./03-report-visualization.md)
