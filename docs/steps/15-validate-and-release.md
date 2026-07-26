# Step 15 — Validate and release

← [14 Export](./14-export-gguf.md) · [Index](./README.md)

---

## Goal

Prove the candidate is good enough on data the **search never saw**, then ship GGUF + recipe + report — or send a concrete constraint back to the optimizer.

---

## Why it exists

A recipe that only looks good on search text may be overfit. Held-out KLD, smoke tests, and benchmarks catch that.

---

## Inputs / Outputs

| | |
|---|---|
| **Input** | Candidate GGUF(s), `logits-heldout.bin`, smoke/bench suites |
| **Output** | Pass/fail + `report.html` / release artifacts — or feedback to Step 13 |

---

## How it works — three tiers

### Tier 1 — Logit fidelity (every candidate · minutes)

```bash
./llama-perplexity -m candidate.gguf \
  --kl-divergence-base logits-heldout.bin \
  --kl-divergence
```

Gates:

- Mean KLD  
- 99.9th-percentile KLD  
- Max KLD (catastrophic single-token failures)  
- Top-token agreement rate  

### Tier 2 — Behavioral smoke (Pareto candidates · ~1 hour)

- Perplexity on an unrelated public set  
- Generation battery: chat, compilable code, exact math, long-context  
- Domain checks from descriptor (e.g. FunctionGemma → schema-valid tool JSON)

### Tier 3 — Benchmarks (final 1–2 · hours)

- MMLU / GSM8K / HumanEval via lm-eval  
- **Statistical** gate: paired vs BF16 inside confidence interval — not a raw score cutoff  

### Feedback loop

```text
Max KLD too high
    → look up which group caused it (sensitivity table)
    → pin that group one level higher
    → re-run optimizer (cache reuses everything else)
```

---

## Example report (sketch)

```text
Model:     functiongemma (resolved BF16 sha 9f2c…)
Recipe:    UD-3.2GB
Size:      3.18 GB

Tier 1 (held-out):
  mean KLD=0.0081  p99.9=0.41  max=2.9  top1=98.7%  PASS

Tier 2:
  code compile 10/10  tool JSON 50/50  PASS

Tier 3:
  MMLU paired Δ=-0.2%  CI95=[-0.9, +0.5]  PASS

→ RELEASE
  functiongemma-UD.gguf
  recipe.yaml
  report.html
```

---

## Done when

- [ ] All required tiers pass  
- [ ] Report published next to recipe  
- [ ] Or failure converted to an optimizer constraint (not a blind restart)

## You finished the pipeline

Return to the [step index](./README.md) or the main [README](../../README.md).
