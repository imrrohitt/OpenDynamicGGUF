# Feature 13 — Security scanner

← [12 Plugin system](./12-plugin-system.md) · [Index](./README.md) · Next: [14 CI/CD GitHub Action](./14-cicd-github-action.md) →

Priority: ⭐⭐⭐⭐☆ · Phase 4 · New modules: `scan.py` · New command: `odg scan`

---

## Goal

Verify that a GGUF is exactly what its recipe claims — nothing more, nothing less:

```bash
odg scan model-UD.gguf --recipe recipe.yaml

  ✓ recipe schema + hashes valid
  ✓ source model provenance: google/gemma-4-27b-it @ 9f2c…
  ✓ per-tensor checksums match reproduced build (312/312)
  ✓ no unexpected tensors / metadata keys
  ✓ quant-type map matches recipe assignments
  PASS — bit-for-bit reproducible from recipe
```

---

## Why it exists

Research ("Mind the Gap", arXiv:2505.23786) demonstrated that malicious behavior can be
hidden specifically in the quantization gap — a model that behaves benignly in full precision
and differently when quantized. Users download GGUF binaries from strangers daily with no
verification story. This project is uniquely positioned to provide one, because
reproducibility is already the core design: **a recipe + source hash is a verifiable claim,
and the scanner is the verifier.** No other GGUF tooling can offer this.

---

## Depends on

- Reproducible export (step 14) — the scanner's ground truth.
- `gguf_tensors.py` — tensor payload access exists already.
- Feature 04/05 — the scanner is what makes marketplace/leaderboard integrity rule 4 real.

---

## Design

### Check tiers (cheap → expensive)

| Tier | Check | Needs |
|---|---|---|
| 1 static | GGUF parses cleanly; metadata keys against allowlist; tensor names/shapes/types match the recipe's assignments; embedded provenance KV consistent with recipe | file only, seconds |
| 2 checksum | per-tensor payload hashes vs a signed manifest (`tensors.sha256` published alongside the GGUF) | manifest, seconds |
| 3 replay | rebuild from recipe (source download + step 14) and compare per-tensor hashes — the full reproducibility proof | source model + compute |

Tier 3 is expensive but needs to run only **once per published artifact** (e.g. in
marketplace/leaderboard CI); everyone else then verifies the cheap tiers against the
now-trusted manifest.

### Suspicion heuristics (report, don't block)

Beyond exact verification, flag anomalies for human review: tensors whose statistics deviate
wildly from the catalog's recorded features, unexpected extra tensors, metadata that
overrides chat templates or sampling defaults. These are warnings with explanations — only
tier 1–3 mismatches are hard failures.

### Trust artifacts

- `tensors.sha256` — per-tensor manifest, generated at export (step 14 addition), uploaded
  by `odg publish`.
- Scan report (`scan.json` + terminal summary) — attached to leaderboard rows.

---

## Build steps

1. **Tier 1 static checks.** GGUF header/metadata/tensor-map validation against a recipe.
   Fixture-based tests with deliberately tampered files.
2. **Per-tensor manifest at export.** Step 14 writes `tensors.sha256`; `odg publish` uploads
   it.
3. **Tier 2 checksum verify.** Stream tensor payloads, compare to manifest.
4. **Tier 3 replay.** Orchestrate rebuild-and-compare through the existing run store
   (resumable — it's just steps).
5. **Suspicion heuristics.** Feature-deviation warnings using catalog stats; documented
   false-positive expectations.
6. **Ecosystem wiring.** Marketplace/leaderboard CI runs tier 3 once per submission; rows
   display scan status.
7. **Threat-model doc.** What the scanner does and does not protect against (e.g. it
   verifies *provenance*, not that the source model itself is safe).

---

## Done when

- [ ] Tampering with one tensor payload, one metadata key, or one quant type is detected by
      the appropriate tier (adversarial test fixtures)
- [ ] `odg scan` tier 1+2 completes in seconds on a 10 GB file
- [ ] Tier 3 replay proves bit-for-bit reproduction for a published model
- [ ] Marketplace/leaderboard submissions carry a scan status
- [ ] Threat model documented honestly — including what is out of scope

## Next

[Feature 14 — CI/CD GitHub Action](./14-cicd-github-action.md)
