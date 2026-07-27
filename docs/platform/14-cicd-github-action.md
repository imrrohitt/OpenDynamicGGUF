# Feature 14 — CI/CD for GGUF (GitHub Action)

← [13 Security scanner](./13-security-scanner.md) · [Index](./README.md) · Next: [15 Web UI](./15-web-ui.md) →

Priority: ⭐⭐⭐⭐☆ · Phase 5 · New tree: `action/` · Consumers: model authors' repos

---

## Goal

Model authors get optimized, benchmarked, documented GGUF releases automatically:

```yaml
# .github/workflows/release-gguf.yml in a model author's repo
on:
  push: {tags: ["v*"]}
jobs:
  gguf:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: opendynamicgguf/action@v1
        with:
          model: ./checkpoint            # or an HF repo id
          device-targets: "rtx-3060-12gb,mac-16gb"
          workload: chat
          suite: standard
          hf-repo: me/my-model-GGUF
          hf-token: ${{ secrets.HF_TOKEN }}
```

Push a model → optimize → benchmark → scan → publish GGUF + recipe + report + card.

---

## Why it exists

Fine-tune authors rarely ship good quants — it's manual, hardware-dependent work outside
their release flow. Putting the whole pipeline behind a GitHub Action makes optimized
releases a repo config line, and every model shipped this way spreads the recipe/report
formats further (the platform's real distribution strategy).

---

## Depends on

- Everything the action orchestrates: features 01 (device targets), 02, 06 (publish),
  11 (cards), 13 (scan). This is deliberately last-ish: it automates a flow that must first
  work interactively.

---

## Design

- **The action is a thin client** (platform invariant 1): a composite action + container
  image that installs `odg` + llama.cpp and runs `odg publish` with the given inputs. All
  logic stays in the CLI — the action adds only CI ergonomics.
- **Hardware reality.** Sensitivity probing wants a GPU; GitHub-hosted runners are
  CPU-only and slow. Supported modes:
  1. `self-hosted` GPU runner (primary, documented path),
  2. `recipe-only` mode — skip search, build + validate from an existing marketplace recipe
     (cheap enough for hosted runners),
  3. `probe-budget` knob to cap cost for small models on hosted runners.
- **Caching is the make-or-break.** The run store maps onto `actions/cache` keyed by
  content hashes (which is exactly how the store already keys artifacts): re-releases of a
  fine-tune reuse corpus, imatrix, and unchanged probes.
- **Fail like CI should.** Gate failures fail the workflow with the gate table in the job
  summary; `report.html` and logs are uploaded as workflow artifacts either way.

---

## Build steps

1. **Container image.** `odg` + pinned llama.cpp build + harness; published to GHCR with
   version tags.
2. **Composite action MVP.** Inputs → `odg publish` invocation; recipe-only mode first
   (works on hosted runners, fastest to demo).
3. **Run-store caching.** Map `artifacts/` into `actions/cache` with content-hash keys;
   verify a re-run skips completed steps.
4. **Full-pipeline mode.** Self-hosted GPU documentation + probe-budget knob for hosted
   runners.
5. **Job summary + artifacts.** Gate table in `$GITHUB_STEP_SUMMARY`; report/card/logs as
   artifacts.
6. **Dogfood.** This repo uses the action to publish its own seed models; the example
   workflow in the README is the one we actually run.

---

## Done when

- [ ] `uses: opendynamicgguf/action@v1` in a fresh repo publishes a complete HF repo from a
      tag push (recipe-only mode, hosted runner)
- [ ] Full pipeline documented and verified on a self-hosted GPU runner
- [ ] Second run with unchanged inputs is mostly cache hits
- [ ] Gate failure = red workflow with a readable gate table, artifacts still uploaded
- [ ] The action contains no logic that isn't reachable from the CLI

## Next

[Feature 15 — Web UI](./15-web-ui.md)
