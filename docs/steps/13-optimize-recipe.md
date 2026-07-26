# Step 13 — Optimize the recipe under a size budget

← [12 Probe](./12-sensitivity-probe.md) · [Index](./README.md) · Next: [14 Export](./14-export-gguf.md) →

---

## Goal

Using the sensitivity table, assign a quantization type to every group so size stays under budget while maximizing bytes saved per unit ΔKLD.

Emit `recipe.yaml`, `recipe.tt`, and a **Pareto set** of alternatives.

---

## Command

```bash
odg optimize --model functiongemma:latest
odg optimize --model functiongemma:latest --budget-mb 180 --force
```

Requires Step 12.

| Flag | Meaning |
|---|---|
| `--budget-mb` | Absolute target size (MiB) |
| `--budget-ratio` | Default `0.72` of all-Q6_K estimate when mb omitted |
| `--no-pins` | Disable embd/lm_head→Q8 and attn_v→Q5 floors |

---

## Algorithm (v1)

1. Start all quantizable groups at Q6_K (respecting role pins).  
2. Greedily apply the downgrade with best `Δbytes / ΔKLD` until size ≤ budget.  
3. Emit Pareto recipes at several budget ratios.

---

## Outputs

```text
steps/13_optimize/
  recipe.yaml              # primary odg/recipe/v1
  recipe.tt                # llama-quantize --tensor-type-file
  pareto/*.yaml            # frontier alternatives
  optimize_manifest.json
  output.json
  status.json
  log.txt
```

---

## Done when

- [x] ≥1 recipe written with per-group assignments
- [x] Traceable to sensitivity rows
- [x] Pareto alternatives saved

## Next

[Step 14 — Export the GGUF](./14-export-gguf.md)
