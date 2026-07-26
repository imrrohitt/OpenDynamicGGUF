# Step 08 — Compute activation features

← [07 Corpus](./07-build-calibration-corpus.md) · [Index](./README.md) · Next: [09 Freeze GGUF](./09-freeze-bf16-gguf.md) →

---

## Goal

Run **calibration text** through the model (or a documented proxy) and record **activation** statistics per tensor / group for probe ranking.

---

## Command

```bash
odg activation-features --model functiongemma:latest
# force proxy (no torch needed):
odg activation-features --model functiongemma:latest --mode proxy
# require real BF16 hooks (needs HF weights + torch + transformers):
odg activation-features --model functiongemma:latest --mode forward --force
```

Requires Steps 06–07 done.

---

## Modes

| Mode | When |
|---|---|
| `auto` (default) | Use forward hooks if BF16 HF path + torch/transformers exist; else `proxy_from_weights` |
| `forward` | Fail if a real forward pass cannot run |
| `proxy` | Always estimate from weight_features + role/depth priors |

Current Ollama Q8 runs use **proxy** until you resolve with `--prefer-hf` and install torch.

---

## Outputs

```text
steps/08_activation_features/
  output.json
  tensor_catalog.json          # activation_features filled
  activation_features.json     # group ranking
  status.json
  log.txt
```

Per tensor:

```json
"activation_features": {
  "range_min": -4.2,
  "range_max": 15.2,
  "absmax": 15.2,
  "outlier_ratio": 0.003,
  "method": "proxy_from_weights"
}
```

---

## Done when

- [x] Calib loaded (docs / token estimate logged)
- [x] Activation features written for tensors / quantizable groups
- [x] Catalog updated / re-hashed

## Next

[Step 09 — Freeze BF16 GGUF](./09-freeze-bf16-gguf.md)
