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
  release/ or release_provisional/
  feedback.json          # on FAIL
  output.json
```

---

## Done when

- [x] Report written
- [x] Release staged **or** optimizer feedback produced

## Pipeline complete

Return to the [step index](./README.md).
