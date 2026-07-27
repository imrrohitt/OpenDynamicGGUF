# Feature 04 — Recipe marketplace

← [03 HTML report](./03-report-visualization.md) · [Index](./README.md) · Next: [05 Public leaderboard](./05-public-leaderboard.md) →

Priority: ⭐⭐⭐⭐⭐ · Phase 2 · New modules: `registry.py`, `recipes/` tree · New commands: `odg recipe …`

---

## Goal

An open, community-driven repository of optimized quantization recipes:

```text
recipes/
  gemma4-27b/
    q4-24gb.yaml
    q4-16gb.yaml
    rtx-3060.yaml
    mac-16gb.yaml
    coding.yaml
    reasoning.yaml
  qwen3/
    …
```

so users can skip the search entirely:

```bash
odg recipe gemma4-27b --device macbook-air-16gb    # fetch + build best match
odg recipe qwen3 --workload coding
odg recipe list gemma4-27b                         # what exists, with measured results
odg recipe submit ./my-recipe.yaml                 # validate before opening a PR
```

---

## Why it exists

This is the biggest ecosystem gap: everyone ships better GGUF *files*, nobody ships shared,
reproducible *decisions*. A recipe is tiny (KB), auditable, and rebuilds the GGUF bit-for-bit
from the source hash — the perfect unit of community exchange. Sensitivity probing is
expensive; once one person has paid that cost for gemma4-27b on a 12 GB card, nobody else
should have to.

---

## Depends on

- Existing recipe format + reproducible export (steps 13–14).
- Feature 01 (device ids used in recipe metadata), feature 02 (measured results embedded).

---

## Design

### Recipe schema v2 (`odg/recipe/v2`)

Everything in v1 stays valid; v2 adds marketplace metadata:

```yaml
schema: odg/recipe/v2
extends: odg/recipe/v1
meta:
  model_family: gemma4-27b
  base_model: google/gemma-4-27b-it        # exact source the hashes refer to
  target_hardware: [rtx-3060-12gb, mac-16gb]
  workload: coding                          # coding|chat|reasoning|math|tools|general
  author: "@handle"
  odg_version: "0.4.0"
  results:                                  # measured, from benchresult.json
    size_gb: 10.8
    mean_kld: 0.0074
    ppl_wikitext: 6.91
    benchresult_sha256: "…"
```

### Registry = git repo, not a service

The marketplace is the `recipes/` tree in this repository (later, optionally, a dedicated
repo). Submission is a pull request. That gives review, history, attribution, and hosting
for free, and CI can machine-check every submission:

1. schema-valid v2,
2. hashes present (source, imatrix, benchresult),
3. `odg export --dry-run` succeeds against the declared base model,
4. declared results match the attached benchresult file.

### Resolution logic (`registry.py`)

`odg recipe <family> --device X --workload Y` scores candidates:

```text
exact device match > compatible budget (recipe size ≤ device budget) ;
exact workload > general ;
newer odg_version > older ; better mean_kld at same size wins ties
```

Prints the match and *why*, then runs the normal build path: fetch base model (resolver) →
export from recipe (step 14) → quick Tier-1 verify (step 15).

---

## Build steps

1. **Schema v2.** Parser + validator, `extends: odg/recipe/v1` handling, round-trip tests.
2. **`recipes/` tree + seed content.** Directory conventions
   (`recipes/<family>/<name>.yaml`), and 2–3 real seeded recipes produced by our own runs.
3. **`odg recipe list`.** Read the local tree, print families/recipes with their measured
   results table.
4. **Resolution scoring.** Pure function, table-driven tests covering device/workload/tie
   cases.
5. **`odg recipe <family> …` build path.** Resolve → confirm plan → resolver + step 14 +
   Tier-1 check. Refuse (with `--allow-mismatch` escape hatch) if the local base model hash
   differs from the recipe's.
6. **`odg recipe submit`.** Local validation of the four CI checks so contributors fix
   issues before the PR.
7. **CI for submissions.** GitHub workflow running the same checks on `recipes/**` PRs.
8. **Remote fetch.** Pull the registry from GitHub at run time (with local cache) so users
   get new recipes without upgrading the package.

---

## Done when

- [ ] `odg recipe gemma4-27b --device rtx-3060` fetches, verifies hash, builds, Tier-1 checks
- [ ] Submission PRs are machine-validated (schema, hashes, dry-run, results consistency)
- [ ] Resolution explains its choice ("matched device exactly; workload fell back to general")
- [ ] Registry tree has ≥3 seeded, measured recipes at launch
- [ ] A recipe author is credited in `meta.author` and surfaced by `list`

## Next

[Feature 05 — Public leaderboard](./05-public-leaderboard.md)
