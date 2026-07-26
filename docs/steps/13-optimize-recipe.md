# Step 13 — Optimize the recipe under a size budget

← [12 Probe](./12-sensitivity-probe.md) · [Index](./README.md) · Next: [14 Export](./14-export-gguf.md) →

---

## Goal

Using the sensitivity table, assign a quantization type to every group so that **quality is maximized under a size/VRAM budget** (equivalently: maximize bytes saved per unit ΔKLD).

Emit a **Pareto set** of recipes, not a single opaque config.

---

## Why it exists

Hand-picking `Q4_K_M` everywhere wastes bits on easy tensors and starves sensitive ones. The optimizer turns measurements into an assignable recipe.

---

## Inputs / Outputs

| | |
|---|---|
| **Input** | Sensitivity table + target size (e.g. 3.2 GB) |
| **Output** | `recipe.yaml` candidates (Pareto frontier) |

---

## How it works

### Objective

```text
maximize   Σ bytes_saved(group, quant)
subject to Σ size(group, quant) ≤ budget
           (proxy: maximize bytes_saved / ΔKLD greedily)
```

### Algorithm (v1)

1. **Start** all groups at a safe high type (e.g. Q6_K / Q8 for embd & lm_head).  
2. **Greedy:** repeatedly apply the downgrade with best `Δbytes / ΔKLD` until budget met.  
3. **Refine:** try ±1 level swaps with *joint* re-measure (full candidate KLD on search).  
4. **Emit** top frontier points (smaller/faster vs higher fidelity).

Skip Bayesian / evolutionary search in v1 — each eval is expensive; greedy + local search is enough.

### Pins (defaults, still overridable by probes)

```text
embedding, lm_head     → Q8_0
attn_v (often)         → ≥ Q5_K / Q6_K
norm                   → skip (F16)
router (MoE)           → pin high
```

---

## Example

Budget: **3.2 GB**

```text
After greedy:
  embedding, output     Q8_0
  attn_v @ all          Q6_K
  attn_q/k early        Q5_K
  attn_q/k mid/late     Q4_K
  ffn_gate/up mid       Q3_K
  ffn_down              Q4_K
  estimated size        3.18 GB
  predicted mean ΔKLD   0.008
```

Write `recipe.yaml` (see main README recipe section for full schema).

---

## Done when

- [ ] ≥1 recipe meets budget
- [ ] Assignments traceable to sensitivity rows
- [ ] Pareto alternatives saved for the user

## Next

[Step 14 — Export the GGUF](./14-export-gguf.md)
