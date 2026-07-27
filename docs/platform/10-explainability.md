# Feature 10 — Explainability

← [09 Multi-objective optimization](./09-multi-objective-optimization.md) · [Index](./README.md) · Next: [11 Automatic model cards](./11-automatic-model-cards.md) →

Priority: ⭐⭐⭐⭐⭐ · Phase 3 (quick win — can ship any time) · New modules: `explain.py` · New command: `odg explain`

---

## Goal

After optimization, explain every decision in plain language:

```text
$ odg explain --model gemma4-27b

token_embd            → Q8_0   pinned: touched by every token; probe Q4_K cost
                               ΔKLD +0.055 for only 190 MB saved
attn_v (all layers)   → Q6_K   pinned: probe Q4_K showed ΔKLD +0.037 for 45 MB —
                               worst bytes-per-quality trade in the table
ffn_up (mid layers)   → Q3_K   cheap bits: 310 MB saved for ΔKLD +0.004
ffn_down (mid layers) → Q4_K   Q3_K probe rejected: ΔKLD +0.019 exceeded the
                               marginal-rate cutoff at this budget
```

---

## Why it exists

Trust in the optimizer is the product. The data for every explanation **already exists** —
the sensitivity table maps `(group, quant) → (Δbytes, ΔKLD)` and the optimizer log records
each greedy acceptance/rejection. This feature is a renderer, not new measurement, which is
why it's the cheapest five-star feature in the list.

---

## Depends on

- Step 12 (sensitivity table) and step 13 (recipe + optimizer decisions). Nothing else.

---

## Design

### Explanation = decision record, rendered

For each group, `explain.py` joins three artifacts:

1. **Final assignment** — from `recipe.yaml`.
2. **Probe evidence** — the sensitivity rows for that group (what was tried, what it cost).
3. **Optimizer action** — why the greedy loop stopped where it did: accepted downgrade
   (best ratio at that iteration), rejected downgrade (worse than the marginal cutoff),
   or pin (role policy, with the probe row showing the pin was justified).

To make (3) exact rather than reconstructed, step 13 gains a small addition: an
`decisions.jsonl` log — one line per considered move
(`group, from, to, dbytes, dkld, ratio, accepted, reason`). Everything else is rendering.

### Outputs

- `odg explain` — terminal table (colored by `ui.py` conventions).
- `explain.json` — structured, consumed by report (03) allocation section and model
  cards (11).
- `--group ffn_down` — deep-dive on one group: all probe rows, its acceptance history,
  and what budget change would flip the decision ("at ≥ 11.2 GB budget, ffn_down stays Q5_K").

### Honesty rules

- Never invent a reason: if a group was never probed at the relevant level (e.g. assigned by
  role default), say "role default, not individually probed".
- Every sentence carries the number it came from; numbers carry the artifact hash.

---

## Build steps

1. **Decision log in step 13.** Emit `decisions.jsonl` from the greedy/refinement loop.
   Tiny diff, no behavior change.
2. **Joiner.** recipe + sensitivity + decisions → per-group explanation records
   (`explain.json`).
3. **Terminal renderer.** `odg explain` table + `--group` deep-dive.
4. **Budget counterfactuals.** From the greedy ordering, compute the flip threshold per
   group ("would stay Q5_K above N GB").
5. **Integrations.** Report allocation section and model-card "why these bits" section
   consume `explain.json`.

---

## Done when

- [ ] Every group in the recipe has an explanation backed by a probe row or an explicit
      "role default" statement — no fabricated reasons
- [ ] `odg explain` works on any finished run, including old runs (graceful without
      `decisions.jsonl`: falls back to table-derived reasoning, labeled as such)
- [ ] `explain.json` consumed by report.html allocation section
- [ ] Deep-dive shows the budget threshold that would change a group's assignment

## Next

[Feature 11 — Automatic model cards](./11-automatic-model-cards.md)
