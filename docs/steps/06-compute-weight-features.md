# Step 06 — Compute weight features

← [05 Catalog](./05-build-tensor-catalog.md) · [Index](./README.md) · Next: [07 Calibration corpus](./07-build-calibration-corpus.md) →

---

## Goal

From each **weight tensor alone**, compute statistics used later for **ranking** (not final bit decisions).

No calibration text is required in this step.

---

## Why it exists

Weight features are cheap and always available. They answer: “Does this matrix *look* easy or hard to quantize?” They do **not** answer: “What happens to the model if we actually quantize it?” That needs Steps 08 and 12.

---

## Inputs / Outputs

| | |
|---|---|
| **Input** | Model tensors + catalog |
| **Output** | `weight_features` filled per tensor / group |

---

## How it works

```text
Weight tensor W
      │
      ▼
mean, variance, entropy
sparsity (% near zero)
outlier_ratio (% beyond e.g. 6σ or percentile)
weight_norm (Frobenius / L2)
spectral_norm (approx: power iteration OK for large mats)
```

```python
import torch

@torch.no_grad()
def weight_features(w: torch.Tensor) -> dict:
    x = w.float().reshape(-1)
    mean = x.mean().item()
    var = x.var(unbiased=False).item()
    std = var ** 0.5 + 1e-12
    sparsity = (x.abs() < 1e-3).float().mean().item()
    outlier_ratio = (x.abs() > 6 * std).float().mean().item()
    weight_norm = torch.linalg.vector_norm(x).item()
    # spectral: optional power iteration on 2D weights
    return {
        "mean": mean,
        "variance": var,
        "sparsity": sparsity,
        "outlier_ratio": outlier_ratio,
        "weight_norm": weight_norm,
        # "entropy": …, "spectral_norm": …
    }
```

Aggregate to **group** level (mean / max of member features) for ranking.

---

## Example

```text
layer12.q_proj.weight
  mean           =  0.001
  variance       =  0.05
  sparsity       =  72%
  outlier_ratio  =  0.2%
  weight_norm    = 13.8
  spectral_norm  =  2.9
```

Interpretation (heuristic only):

```text
High sparsity + low variance + few outliers  →  probably easy → try Q3 first
Many outliers + large spectral norm          →  probably hard → start at Q5/Q6
```

**Critical reminder:** two tensors can share identical weight stats and still have very different ΔKLD after quantization. Features only prioritize probe order.

---

## Done when

- [ ] Every quantizable tensor has `weight_features`
- [ ] Group-level aggregates stored in catalog
- [ ] Catalog hash updated

## Next

[Step 07 — Build the calibration corpus](./07-build-calibration-corpus.md)
