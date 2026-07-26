# Step 15 — Validate and release

← [14 Export](./14-export-gguf.md) · [Index](./README.md)

---

## Goal

Gate the candidate on held-out criteria, write a report, and stage a release — or return feedback to the optimizer.

---

## Command

```bash
odg validate --model functiongemma:latest
odg validate --model functiongemma:latest --strict   # FAIL if no real GGUF
```

Requires Step 14.

Verdicts:
- `RELEASE` — candidate GGUF present + tiers pass
- `PROVISIONAL` — dry-run export OK (plumbing); re-export with llama for release
- `FAIL` — gates failed; see `feedback.json`

---

## Outputs

```text
steps/15_validate/
  report.md / report.html
  quantization_report_card.html   # full per-layer / per-group card
  quantization_report_card.md
  quantization_report_card.json
  release/ or release_provisional/   # copies of the above
  feedback.json          # on FAIL
  output.json
```

The **Quantization Report Card** includes: architecture (layers/tensors), size vs baseline, compression by role, a per-layer matrix (attn_q/k/v/o + ffn_*), and per-group Δbytes / ΔKLD.

---

## Done when

- [x] Report written
- [x] Release staged **or** optimizer feedback produced

## Pipeline complete

Return to the [step index](./README.md).
