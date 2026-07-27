# OpenDynamicGGUF — Platform Architecture

**From a quantization tool to an open optimization platform for local LLM deployment.**

This document describes the target architecture that grows the existing 15-step pipeline
(`resolve` → `validate`) into a full platform. The per-feature breakdowns — each feature split
into small, shippable steps — live in [`docs/platform/`](docs/platform/README.md).

The existing pipeline docs are unchanged and remain the core: [`docs/steps/`](docs/steps/README.md).

---

## 1. The shift

Today the project answers one question:

```text
BF16 model + size budget  →  best per-tensor GGUF
```

The platform answers the question users actually ask:

```text
"I have this hardware and this workload — give me the best model, prove it, and let me share it."
```

Pipeline evolution:

```text
Today                          Platform
─────                          ────────
BF16                           Model (any ref, any source)
 ↓                              ↓
GGUF                           Analyze     (catalog, features, sensitivity)
                                ↓
                               Search      (recipes, AutoML, multi-objective)
                                ↓
                               Optimize    (hardware-aware bit allocation)
                                ↓
                               Validate    (benchmarks, gates, security scan)
                                ↓
                               Publish     (reports, model cards, HF, leaderboard)
```

Everything already built (steps 01–15) becomes the **engine** of this platform. Nothing is
thrown away; new layers are wrapped around it.

---

## 2. Layered architecture

```text
┌─────────────────────────────────────────────────────────────────────┐
│  INTERFACE LAYER                                                     │
│  cli.py (odg)  ·  Web UI  ·  GitHub Action (CI/CD)                   │
├─────────────────────────────────────────────────────────────────────┤
│  DISTRIBUTION LAYER                                                  │
│  Recipe marketplace  ·  Public leaderboard  ·  HF publish  ·  Cards  │
├─────────────────────────────────────────────────────────────────────┤
│  TRUST LAYER                                                         │
│  HTML reports  ·  Explainability  ·  Security scanner  ·  Benchmarks │
├─────────────────────────────────────────────────────────────────────┤
│  DECISION LAYER                                                      │
│  Hardware profiles  ·  Multi-objective optimizer  ·  Recipe search   │
│  (AutoML)  ·  Calibration dataset builder                            │
├─────────────────────────────────────────────────────────────────────┤
│  CORE ENGINE (exists today — steps 01–15)                            │
│  resolve · load · enumerate · classify · catalog · features ·        │
│  corpus · freeze · imatrix · logits · sensitivity · optimize ·       │
│  export · validate  — checkpointed run store (store.py / steps.py)   │
├─────────────────────────────────────────────────────────────────────┤
│  EXTENSION LAYER (cross-cutting)                                     │
│  Plugin system: metrics / search algorithms / exporters / eval tasks │
└─────────────────────────────────────────────────────────────────────┘
```

### Layer responsibilities

| Layer | Responsibility | Rule |
|---|---|---|
| Core engine | Measure, search, export, gate. Content-addressed, resumable. | Never changes contract: JSON artifacts in the run store |
| Decision | Translate *user intent* (hardware, workload, trade-offs) into engine inputs (budgets, pins, objectives) | Users say "16 GB MacBook", never "3.2 GB budget" |
| Trust | Make every decision inspectable: reports, reasons, benchmarks, checksums | No number without a source artifact |
| Distribution | Move recipes and results between users: marketplace, leaderboard, HF | Recipes are the unit of sharing, not GGUF files |
| Interface | CLI first; Web UI and CI are thin clients over the same engine | No logic lives in an interface |
| Extension | Third-party metrics/search/exporters/evals without touching core | Plugins register through entry points, core never imports them |

---

## 3. New subsystems and where they live

Flat-module convention is kept. New modules at repo root, grouped by prefix where a
subsystem has several files:

```text
OpenDynamicGGUF/
├── (existing 15-step modules …)
│
│  DECISION LAYER
├── hardware.py             # hardware profile DB + detection (--gpu/--ram/--device)
├── objectives.py           # multi-objective scoring: quality/speed/vram/ram/latency
├── search_space.py         # candidate recipe generation for AutoML search
├── autosearch.py           # search driver: random → evolutionary → Pareto keep
├── calib_builder.py        # domain calibration dataset builder (coding/chat/math/…)
│
│  TRUST LAYER
├── report.py               # report.html generator (charts from run artifacts)
├── explain.py              # per-group decision explanations from sensitivity table
├── benchmark.py            # lm-eval-harness runner + benchmarks.html
├── scan.py                 # security scanner: checksums, provenance, recipe replay
│
│  DISTRIBUTION LAYER
├── recipes/                # in-repo recipe registry (community marketplace)
│   └── <model-family>/<recipe>.yaml
├── registry.py             # recipe resolution: name+device/workload → recipe.yaml
├── publish.py              # HF Hub: download BF16 → optimize → upload GGUF+recipe+report
├── modelcard.py            # auto model card generator (README.md for HF)
├── leaderboard/            # static-site generator + results data (separate concern)
│
│  EXTENSION LAYER
├── plugins.py              # entry-point discovery: odg.metrics / odg.search / odg.export / odg.evals
│
│  INTERFACE LAYER
├── cli.py                  # extended: odg fit / recipe / benchmark / report / publish / scan / card
├── webui/                  # local FastAPI + static frontend over the run store
└── action/                 # GitHub Action wrapper (composite action + Dockerfile)
```

---

## 4. New CLI surface

The engine keeps its step commands. The platform adds **intent-level** commands:

```bash
# Decision layer — users describe hardware/workload, not bits
odg fit --model gemma4-27b --gpu 24GB                 # hardware-aware optimize
odg fit --model qwen3 --device macbook-air-16gb --workload coding

# Distribution layer — recipes are first-class
odg recipe list gemma4-27b
odg recipe gemma4-27b --device rtx-3060               # fetch + build from registry
odg recipe submit ./recipe.yaml                       # validate for marketplace PR

# Trust layer
odg benchmark model.gguf --suite standard             # MMLU/GSM8K/HumanEval/…
odg report --model gemma4-27b                         # report.html from run artifacts
odg explain --model gemma4-27b                        # why each group got its bits
odg scan model.gguf --recipe recipe.yaml              # provenance + checksum verify

# Publish layer
odg publish --model unsloth/Qwen3.6 --gpu 16GB --hf-repo me/Qwen3.6-UD
odg card --model gemma4-27b                           # model card only
```

---

## 5. Key data contracts

The platform is held together by four schemas. All are versioned (`odg/<kind>/v1`) and
content-addressed like existing artifacts.

### 5.1 Hardware profile (`odg/hardware/v1`)

```yaml
schema: odg/hardware/v1
id: rtx-3060-12gb
kind: gpu                    # gpu | apple_silicon | cpu
vram_gb: 12
ram_gb: 32
bandwidth_gbps: 360
usable_fraction: 0.90        # leave headroom for KV cache + runtime
kv_cache_reserve_gb: 1.5     # scaled by context target
```

`odg fit` converts a profile + context length into the byte budget the existing optimizer
(step 13) already accepts. That is the whole trick: **hardware awareness is a front-end to
the current knapsack, not a new optimizer.**

### 5.2 Recipe (extends existing `odg/recipe/v1`)

Marketplace recipes add metadata on top of the current format:

```yaml
schema: odg/recipe/v2
extends: odg/recipe/v1        # everything in v1 stays valid
meta:
  model_family: gemma4-27b
  target_hardware: [rtx-3060-12gb, mac-16gb]
  workload: coding            # coding | chat | reasoning | math | tools | general
  author: "@handle"
  results:                    # measured, links to run artifacts
    size_gb: 10.8
    mean_kld: 0.0074
    ppl_wikitext: 6.91
    toks_per_sec: {rtx-3060-12gb: 34.2}
```

### 5.3 Benchmark result (`odg/benchresult/v1`)

One JSON per (gguf, suite) pair; consumed by reports, model cards, and the leaderboard.
Always stores the **paired delta vs BF16** with confidence intervals, never just raw scores
(design principle 7).

### 5.4 Plugin entry points

```toml
# a third-party package's pyproject.toml
[project.entry-points."odg.metrics"]
my_metric = "my_pkg.metric:MyMetric"

[project.entry-points."odg.search"]
anneal = "my_pkg.search:SimulatedAnnealing"

[project.entry-points."odg.evals"]
sql_tasks = "my_pkg.evals:SqlEvalSuite"
```

Core defines small ABCs (`Metric`, `SearchAlgorithm`, `Exporter`, `EvalSuite`) in
`plugins.py`; built-in implementations register through the same mechanism so the plugin
path is exercised constantly, not only by third parties.

---

## 6. How the features compose (one flow)

`odg publish --model unsloth/Qwen3.6 --gpu 16GB --workload coding --hf-repo me/qwen3.6-ud`

```text
hardware.py      16GB GPU + ctx target        →  byte budget + KV reserve
calib_builder.py workload=coding              →  coding-heavy calib corpus (step 07 input)
core engine      steps 01–12                  →  sensitivity table
autosearch.py    candidates around greedy     →  Pareto frontier of recipes
objectives.py    quality × speed × vram       →  pick frontier point for this profile
core engine      steps 13–15                  →  GGUF + gates
benchmark.py     suite=standard               →  benchresult.json (paired vs BF16)
explain.py       sensitivity + recipe         →  per-group reasons
report.py        all artifacts                →  report.html
scan.py          recipe replay + checksums    →  provenance block
modelcard.py     everything above            →  README.md
publish.py       HF Hub                       →  GGUF + recipe + report + card uploaded
leaderboard      benchresult.json PR          →  public comparison row
```

Every arrow is a checkpointed step in the existing run store — resumable, cached,
content-addressed.

---

## 7. Build order

Priorities (detail and step-by-step breakdowns in [`docs/platform/`](docs/platform/README.md)):

| Phase | Features | Why this order |
|---|---|---|
| **P1 — Decide** | [01 Hardware-aware optimizer](docs/platform/01-hardware-aware-optimizer.md) · [02 Benchmark runner](docs/platform/02-benchmark-runner.md) · [03 HTML report](docs/platform/03-report-visualization.md) | Biggest usability win; both feed everything downstream |
| **P2 — Share** | [04 Recipe marketplace](docs/platform/04-recipe-marketplace.md) · [05 Leaderboard](docs/platform/05-public-leaderboard.md) · [06 HF integration](docs/platform/06-huggingface-integration.md) | Turns users into contributors; needs P1 outputs to have something to share |
| **P3 — Deepen** | [07 Calibration builder](docs/platform/07-calibration-dataset-builder.md) · [08 Recipe search / AutoML](docs/platform/08-recipe-search-automl.md) · [09 Multi-objective](docs/platform/09-multi-objective-optimization.md) · [10 Explainability](docs/platform/10-explainability.md) | Better answers, not just easier answers |
| **P4 — Harden** | [11 Model cards](docs/platform/11-automatic-model-cards.md) · [12 Plugin system](docs/platform/12-plugin-system.md) · [13 Security scanner](docs/platform/13-security-scanner.md) | Trust + extensibility for ecosystem growth |
| **P5 — Broaden** | [14 CI/CD action](docs/platform/14-cicd-github-action.md) · [15 Web UI](docs/platform/15-web-ui.md) | New audiences on a now-stable engine |

Dependency graph between features:

```text
01 hardware ─────────┐
02 benchmark ──┬─────┼──► 05 leaderboard
03 report ─────┤     ├──► 06 hf publish ──► 11 model cards ──► 14 ci/cd
               │     │
07 calib ──────┤     └──► 04 marketplace
08 automl ◄────┘
09 multi-objective ◄── 01 + 08
10 explain ◄── existing sensitivity table (can ship almost immediately)
12 plugins — cross-cutting, introduce interfaces early, enforce in P4
13 scanner ◄── recipe reproducibility (exists) + checksums
15 web ui ◄── everything (thin client)
```

---

## 8. Platform invariants

These extend the existing design principles and every new feature must respect them:

1. **The engine stays headless.** Web UI, CI action, and marketplace are clients of the same
   run store and CLI. No optimization logic outside the core modules.
2. **Recipes are the unit of exchange.** The marketplace and leaderboard share recipes and
   measured results — GGUF binaries are always rebuildable from recipe + source hash.
3. **Intent in, budget out.** Users express hardware and workload; only the decision layer
   translates to bytes/pins. Engine inputs never grow user-facing vocabulary.
4. **No unmeasured claim leaves the system.** Reports, cards, and leaderboard rows link every
   number to a run artifact hash.
5. **Paired statistics or nothing.** All published comparisons are paired vs the BF16
   reference with confidence intervals (existing gate rule, now platform-wide).
6. **Plugins extend; forks are a bug.** If a contributor must patch core to add a metric,
   search strategy, exporter, or eval, the plugin API is missing a hook.
