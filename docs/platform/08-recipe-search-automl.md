# Feature 08 — Recipe search (AutoML)

← [07 Calibration dataset builder](./07-calibration-dataset-builder.md) · [Index](./README.md) · Next: [09 Multi-objective optimization](./09-multi-objective-optimization.md) →

Priority: ⭐⭐⭐⭐⭐ · Phase 3 · New modules: `search_space.py`, `autosearch.py` · CLI: `odg optimize --search …`

---

## Goal

Go beyond one greedy answer:

```text
Current:   sensitivity table → greedy → 1 recipe (+ fixed-ratio Pareto points)

Search:    sensitivity table → generate many candidate recipes
                             → score cheaply (additive model)
                             → measure the promising ones (real joint KLD)
                             → return a *measured* Pareto frontier
```

```bash
odg optimize --model gemma4 --search evolutionary --eval-budget 40
```

This is neural-architecture-search methodology applied to bit allocation.

---

## Why it exists

Greedy + local refinement (step 13) assumes per-group effects are additive. That's ~90%
right, but interaction effects are real — especially in MoE and hybrid models — and the
remaining 10% is exactly where an open project can beat closed heuristics. The core README
explicitly deferred this ("later experiment, not a prerequisite"); this feature is that
experiment, structured.

---

## Depends on

- Step 12's sensitivity table (the cheap surrogate) and step 13's greedy (the seed and the
  baseline to beat).
- Cache discipline (design principle 6): every candidate eval is a quantize + Tier-1 pass,
  content-addressed so repeated configs are free.
- Feature 12's `odg.search` plugin interface (this feature defines it; plugins add more
  algorithms later).

---

## Design

### Two-tier evaluation (the cost trick)

| Tier | Evaluator | Cost | Used for |
|---|---|---|---|
| surrogate | additive ΔKLD sum from sensitivity table + exact Δbytes | free | scoring thousands of candidates |
| measured | real export (step 14, can be `--dry-run` sized first) + Tier-1 KLD on the **search split** | ~5–15 min | top-K per generation, `--eval-budget` total |

The held-out split stays untouched (design principle 4); final frontier candidates go
through the normal step 15 gates.

### Search space (`search_space.py`)

A candidate = mapping `group → quant_type`, constrained by:

- role pins (embd/output/router floors — same pins step 13 uses),
- monotone depth hints (optional: early layers ≥ late layers for attn),
- budget window (candidates outside ±10% of target are rejected before scoring).

### Driver (`autosearch.py`)

```text
seed population   = greedy result + its Pareto neighbors + random perturbations
loop generations:
  mutate (±1 level on 1–3 groups) + crossover
  surrogate-score all → select top-K
  measure top-K jointly (real KLD)          ← consumes eval budget
  update: measured points also correct the surrogate (learned interaction offsets)
return: non-dominated set (size, measured KLD)
```

Algorithms behind one interface: `random`, `evolutionary` (default), plugins later.

---

## Build steps

1. **Candidate + constraint model.** `search_space.py`: encode/validate/mutate candidates,
   exact byte accounting. Pure, heavily unit-tested.
2. **Surrogate scorer.** Additive ΔKLD from the sensitivity table. Verify it reproduces
   step 13's greedy ordering exactly (regression test).
3. **Measured evaluator.** Wrap export + Tier-1 into one cached `evaluate(candidate)` call
   keyed by candidate hash.
4. **Random search MVP.** Random valid candidates, surrogate filter, measure top-K, emit
   frontier. Establishes the harness and the baseline.
5. **Evolutionary driver.** Mutation/crossover/selection over generations under
   `--eval-budget`.
6. **Surrogate correction.** Feed measured-vs-predicted residuals back as per-pair
   interaction offsets; report surrogate error in the log.
7. **CLI + artifacts.** `--search`, `--eval-budget`, `--generations`; frontier written like
   step 13's `pareto/` so report (03) and marketplace (04) consume it unchanged.
8. **Benchmark the feature itself.** One published comparison: greedy vs search frontier on
   two models (one dense, one MoE). If search doesn't dominate, say so in the doc.

---

## Done when

- [ ] `--search evolutionary --eval-budget 40` returns a measured, non-dominated frontier
- [ ] Never exceeds the eval budget; repeated candidates hit cache
- [ ] Held-out split provably untouched during search (audit the artifact reads)
- [ ] Greedy remains the default; search is opt-in until the published comparison justifies more
- [ ] Search algorithms load through the same interface plugins will use

## Next

[Feature 09 — Multi-objective optimization](./09-multi-objective-optimization.md)
