# OpenDynamicGGUF — Platform feature breakdowns

This folder breaks the platform expansion (see [`../../ARCHITECTURE.md`](../../ARCHITECTURE.md))
into **small, shippable steps** — the same way [`../steps/`](../steps/README.md) breaks down the
core pipeline.

Each feature file answers the same questions:

| Section | Meaning |
|---|---|
| Goal | What the feature achieves, in one user-visible sentence |
| Why it exists | The user problem / ecosystem gap it closes |
| Depends on | Core steps and other features it builds on |
| Design | Modules, data contracts, CLI surface |
| Build steps | Small ordered steps, each independently mergeable |
| Done when | Checklist before calling the feature shipped |

---

## Big picture

```text
Model → Analyze → Search → Optimize → Validate → Publish
        (steps    (08     (01, 09)   (02, 13)   (03, 04, 05,
         01–12)    automl)                        06, 11)
```

The core engine (steps 01–15) is unchanged. Platform features wrap around it in five phases.

---

## Feature index

### Phase 1 — Decide (highest priority) — ✅ implemented

| # | Feature | File | One-line summary | State |
|---|---|---|---|---|
| 01 | Hardware-aware optimizer | [01-hardware-aware-optimizer.md](./01-hardware-aware-optimizer.md) | `odg fit --gpu 24GB` — hardware in, budget out | ✅ `hardware.py` · `odg fit` / `odg devices` |
| 02 | Benchmark runner | [02-benchmark-runner.md](./02-benchmark-runner.md) | `odg benchmark model.gguf` — paired suites vs BF16 | ✅ `benchmark.py` · `odg benchmark` (llama-bench + optional lm-eval) |
| 03 | Interactive HTML report | [03-report-visualization.md](./03-report-visualization.md) | `report.html` — heatmaps, Pareto, KLD distributions | ✅ `report.py` · `odg report` (auto after `run`/`fit`) |

### Phase 2 — Share

| # | Feature | File | One-line summary |
|---|---|---|---|
| 04 | Recipe marketplace | [04-recipe-marketplace.md](./04-recipe-marketplace.md) | Community recipe registry, `odg recipe <model> --device …` |
| 05 | Public leaderboard | [05-public-leaderboard.md](./05-public-leaderboard.md) | Size / PPL / KLD / tok-s comparison site |
| 06 | Hugging Face integration | [06-huggingface-integration.md](./06-huggingface-integration.md) | `odg publish` — download → optimize → upload, one command |

### Phase 3 — Deepen

| # | Feature | File | One-line summary |
|---|---|---|---|
| 07 | Calibration dataset builder | [07-calibration-dataset-builder.md](./07-calibration-dataset-builder.md) | Open, workload-targeted calib corpora |
| 08 | Recipe search (AutoML) | [08-recipe-search-automl.md](./08-recipe-search-automl.md) | Many candidates → measured Pareto frontier |
| 09 | Multi-objective optimization | [09-multi-objective-optimization.md](./09-multi-objective-optimization.md) | Quality × speed × VRAM × RAM × latency |
| 10 | Explainability | [10-explainability.md](./10-explainability.md) | `odg explain` — why every group got its bits |

### Phase 4 — Harden

| # | Feature | File | One-line summary |
|---|---|---|---|
| 11 | Automatic model cards | [11-automatic-model-cards.md](./11-automatic-model-cards.md) | Complete HF model card from run artifacts |
| 12 | Plugin system | [12-plugin-system.md](./12-plugin-system.md) | Entry-point metrics / search / exporters / evals |
| 13 | Security scanner | [13-security-scanner.md](./13-security-scanner.md) | Provenance, checksums, recipe replay verification |

### Phase 5 — Broaden

| # | Feature | File | One-line summary |
|---|---|---|---|
| 14 | CI/CD GitHub Action | [14-cicd-github-action.md](./14-cicd-github-action.md) | Push model → optimized GGUF release, automatically |
| 15 | Web UI | [15-web-ui.md](./15-web-ui.md) | Browser front-end over the same run store |

---

## Recommended build order

Ship in phase order; within a phase the features are largely parallel. The critical path is:

```text
01 hardware-aware  →  02 benchmark  →  03 report  →  06 hf publish  →  04 marketplace  →  05 leaderboard
```

Feature 10 (explainability) has no new dependencies — the sensitivity table already contains
everything it needs — so it can be picked up at any time as a quick win.

**Rule to remember, platform-wide:** intent in, budget out; recipes are the unit of exchange;
no unmeasured claim leaves the system.
