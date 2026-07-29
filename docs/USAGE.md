# How to use OpenDynamicGGUF

End-to-end guide: install → run the pipeline → read the recipe → **write a real quantized GGUF**.

> **Important:** finishing `odg run` / `odg validate` does **not** always mean a `.gguf` was written.
> If `llama-quantize` is missing, export uses **`dry_run`**: it plans the command and estimates size, but `gguf_out` stays `null`. See [Dry-run vs real export](#dry-run-vs-real-export).

---

## Table of contents

1. [Install](#1-install)
2. [Install llama.cpp (required for a real GGUF)](#2-install-llamacpp-required-for-a-real-gguf)
3. [One-command full pipeline](#3-one-command-full-pipeline)
4. [Step-by-step commands](#4-step-by-step-commands)
5. [Dry-run vs real export](#5-dry-run-vs-real-export)
6. [Write the real quantized GGUF](#6-write-the-real-quantized-gguf)
7. [Reading sizes & the recipe](#7-reading-sizes--the-recipe)
8. [Artifacts layout](#8-artifacts-layout)
9. [Useful flags](#9-useful-flags)
10. [Inspecting runs](#10-inspecting-runs)
11. [Troubleshooting](#11-troubleshooting)
12. [Platform commands: fit, benchmark, report](#12-platform-commands-fit-benchmark-report)

---

## 1. Install

```bash
git clone https://github.com/imrrohitt/OpenDynamicGGUF.git
cd OpenDynamicGGUF

python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Check the CLI:

```bash
odg --help
odg formats          # supported target profiles (q4_k_m, q5_k_m, …)
```

---

## 2. Install llama.cpp (required for a real GGUF)

OpenDynamicGGUF does **not** ship quantization kernels. Real export needs `llama-quantize` (and optionally `llama-imatrix`, `llama-perplexity`, `convert_hf_to_gguf.py`).

### Build (example)

```bash
git clone https://github.com/ggml-org/llama.cpp.git ~/llama.cpp
cd ~/llama.cpp
cmake -B build
cmake --build build --config Release -j
```

You should have something like:

```text
~/llama.cpp/build/bin/llama-quantize
~/llama.cpp/build/bin/llama-imatrix
~/llama.cpp/build/bin/llama-perplexity
```

### Tell `odg` where it is

Pick **one**:

```bash
# preferred
export LLAMA_CPP_DIR=~/llama.cpp

# or alias
export LLAMA_CPP=~/llama.cpp

# or pass the binary per command
odg export --model functiongemma:latest --mode llama \
  --llama-quantize ~/llama.cpp/build/bin/llama-quantize --force
```

`odg` also searches `PATH`, `~/llama.cpp/build/bin/`, Homebrew, etc.

| Tool | Used by |
|---|---|
| `convert_hf_to_gguf.py` | `odg freeze-gguf --mode hf-convert` |
| `llama-imatrix` | `odg imatrix --mode llama` |
| `llama-perplexity` | `odg reference-logits --mode llama`, `odg validate --mode llama` |
| **`llama-quantize`** | **`odg export --mode llama`** (this writes the GGUF) |

Without these binaries, steps still complete using **proxy / dry-run** modes so you can inspect catalogs, recipes, and size estimates.

---

## 3. One-command full pipeline

```bash
# interactive quant-format picker in a TTY, or pass -q
odg run --model functiongemma:latest --quant q4_k_m --no-ask

# fresh run (ignore CURRENT)
odg run --model functiongemma:latest -q q4_k_m --new-run --no-ask

# stop early / resume mid-pipeline
odg run --model functiongemma:latest -q q4_k_m --until optimize --no-ask
odg run --model functiongemma:latest --from-step export --no-ask
```

This runs steps **01 → 15** into a checkpointed run under `artifacts/runs/<run-id>/`.

After it finishes, always check whether export was real:

```bash
odg status --model functiongemma:latest
# then open steps/14_export/output.json → look at "method" and "gguf_out"
```

---

## 4. Step-by-step commands

Same pipeline, one command at a time (useful for debugging). Each step is checkpointed; re-runs are no-ops unless you pass `--force`.

```bash
MODEL=functiongemma:latest

# 01 · any ref → original full-precision / working GGUF + architecture descriptor
odg resolve --model $MODEL --quant q4_k_m --no-ask

# 02–05 · open model → inventory → roles → tensor_catalog.json
odg load        --model $MODEL
odg enumerate   --model $MODEL
odg classify    --model $MODEL
odg catalog     --model $MODEL

# 06 · weight stats (no text needed)
odg weight-features --model $MODEL --only-quantizable

# 07 · calib / search / held-out text (use 300000+ for production)
odg corpus --model $MODEL --target-tokens 300000

# 08 · activation stats (forward if BF16+torch available, else proxy)
odg activation-features --model $MODEL

# 09 · freeze hashed GGUF reference (HF→BF16 if possible, else promote source)
odg freeze-gguf --model $MODEL

# 10–11 · imatrix + reference logits (llama tools if present, else proxy)
odg imatrix           --model $MODEL
odg reference-logits  --model $MODEL

# 12–13 · sensitivity table → recipe.yaml under size budget
odg sensitivity --model $MODEL
odg optimize    --model $MODEL

# 14 · export candidate GGUF (needs llama-quantize for a real file)
odg export --model $MODEL --mode llama --force

# 15 · held-out gates + report / release staging
odg validate --model $MODEL
# refuse PROVISIONAL if there is no real GGUF:
odg validate --model $MODEL --strict
```

Design detail for every step: [`docs/steps/README.md`](./steps/README.md).

---

## 5. Dry-run vs real export

### What you saw

If `steps/15_validate/input.json` looks like:

```json
{
  "export_method": "dry_run",
  "gguf_out": null
}
```

…then **no quantized GGUF was written**. Export only planned the `llama-quantize` command because the binary was missing (or `--mode dry-run` was used).

### How to confirm

```bash
cat artifacts/runs/<run-id>/steps/14_export/output.json
```

| Field | Dry-run | Real export |
|---|---|---|
| `method` | `"dry_run"` | `"llama_quantize"` |
| `gguf_out` | `null` | path to `*-UD.gguf` |
| `gguf_out_nbytes` | `null` | actual file size |
| `estimated_bytes` | from recipe | from recipe (still present) |

Dry-run still writes useful artifacts:

```text
steps/14_export/
  recipe.yaml / recipe.tt
  quantize_command.sh              # exact command that would run
  <model>-UD.gguf.MISSING          # marker — not a GGUF
  export_manifest.json
  output.json
```

Validate then stages a **PROVISIONAL** release (`release_provisional/`) instead of a full `release/`.

> Note: odg’s `"dry_run"` is **not** llama.cpp’s `--dry-run` flag. It means “do not run quantize / no GGUF produced.”

---

## 6. Write the real quantized GGUF

After optimize (step 13) has produced a recipe:

```bash
export LLAMA_CPP_DIR=~/llama.cpp   # or wherever you built it

odg export --model functiongemma:latest --mode llama --force
odg validate --model functiongemma:latest --force
```

- `--mode llama` — **fail hard** if `llama-quantize` is missing (no silent dry-run).
- `--force` — re-run even if the step was already marked done as dry-run.
- Optional: `--llama-quantize /path/to/llama-quantize`

Success looks like:

```text
steps/14_export/
  functiongemma-latest-UD.gguf     # ← the real file
  …
```

and `output.json` has `"method": "llama_quantize"` with a non-null `gguf_out`.

You can also run the saved script yourself:

```bash
bash artifacts/runs/<run-id>/steps/14_export/quantize_command.sh
```

(ensure `llama-quantize` is on `PATH`, or edit the script to use the full binary path).

---

## 7. Reading sizes & the recipe

Example from a `functiongemma:latest` / `q4_k_m` run (estimates only until real export):

| | Size |
|---|---|
| **Estimated quantized** | **~216.2 MiB** (`226,696,704` bytes) |
| Budget target | ~216.3 MiB (`226,773,319` bytes) |
| Source (frozen input GGUF, here Q8_0) | ~287 MiB (`300,796,576` bytes) |
| Target profile | `q4_k_m` |

So the plan is about **~71 MiB smaller** than that source (~25% cut). That number comes from the recipe knapsack estimate — **not** from a written file.

Where to read it:

```bash
# recipe estimate
grep -A4 '^estimate:' \
  artifacts/runs/<run-id>/steps/13_optimize/recipe.yaml

# export copy of the same estimate
python -c "import json; print(json.load(open('artifacts/runs/<run-id>/steps/14_export/output.json'))['estimated_bytes'])"

# after a real export — actual bytes
ls -lh artifacts/runs/<run-id>/steps/14_export/*-UD.gguf
```

Recipe sketch:

```yaml
budget:
  target_size_bytes: 226773319
  target_size_mb: 216.27
base_type: q6_k
overrides:
  "(?:token_embd\\.weight)": q8_0
  "blk\\.(6|7|8|9|10|11)\\.ffn_gate\\.weight": q2_k
  # …
estimate:
  size_bytes: 226696704
  size_mb: 216.19
  predicted_mean_delta_kld: 0.120092
  method: greedy_knapsack_v1
```

`recipe.tt` is the same assignment rendered for `llama-quantize --tensor-type-file`.

---

## 8. Artifacts layout

```text
artifacts/runs/<run-id>/
  CURRENT / meta …          # run bookkeeping
  steps/
    01_resolve/
    02_load/
    …
    09_freeze_gguf/
      model-ref.gguf        # frozen input for quantize
    13_optimize/
      recipe.yaml
      recipe.tt
      pareto/
    14_export/
      *-UD.gguf             # only after real llama export
      *.gguf.MISSING        # dry-run marker
      quantize_command.sh
    15_validate/
      report.md / report.html
      quantization_report_card.*
      release/ or release_provisional/
```

Every step directory also has `input.json`, `output.json`, `status.json`, and `log.txt`.

---

## 9. Useful flags

| Flag | Where | Effect |
|---|---|---|
| `--quant` / `-q` | `resolve`, `run`, `optimize`, `export`, … | Target profile (`q4_k_m`, `q5_k_m`, …). See `odg formats`. |
| `--no-ask` | `resolve`, `run` | Don’t prompt; use `--quant` or default `q4_k_m`. |
| `--new-run` | `resolve`, `run` | Always create a fresh run. |
| `--run <id>` | most commands | Resume / target a specific run. |
| `--force` | most commands | Recompute even if checkpointed `done`. |
| `--prefer-hf` | `resolve`, `run` | Prefer upstream HF BF16 over local Ollama GGUF. |
| `--download-weights` | `resolve`, `run` | Fetch resolved weights. |
| `--mode auto\|llama\|dry-run` | `export` | `auto` falls back to dry-run; `llama` requires the binary. |
| `--mode auto\|llama\|proxy` | `imatrix`, `reference-logits`, `sensitivity`, `validate` | Prefer real llama tools or stay in proxy. |
| `--budget-mb` / `--budget-ratio` | `optimize` | Override the size budget. |
| `--strict` | `validate` | Fail if there is no real GGUF (no PROVISIONAL). |
| `--until` / `--from-step` | `run` | Partial pipeline. |
| `--artifacts` | global | Change artifacts root (default `./artifacts`). |

---

## 10. Inspecting runs

```bash
odg runs
odg status --model functiongemma:latest
odg status --run 20260727-171755-functiongemma-latest
```

Open the human reports:

```bash
open artifacts/runs/<run-id>/steps/15_validate/report.html
open artifacts/runs/<run-id>/steps/15_validate/quantization_report_card.html
```

---

## 11. Troubleshooting

### `export_method: dry_run` / `gguf_out: null`

`llama-quantize` was not found. Install llama.cpp, set `LLAMA_CPP_DIR`, then:

```bash
odg export --model <model> --mode llama --force
odg validate --model <model> --force
```

### Validate verdict is `PROVISIONAL`

Expected after a dry-run export. Re-export with `--mode llama`, then validate again. Use `--strict` if you want validate to fail instead of staging provisional.

### Source GGUF is Q8, not BF16

`odg freeze-gguf` in `auto` mode may **promote** a working Ollama/local GGUF when HF→BF16 conversion isn’t available. For a true BF16 freeze:

```bash
export LLAMA_CPP_DIR=~/llama.cpp
odg freeze-gguf --model <model> --mode hf-convert --require-bf16 --force
```

(Requires a resolved HF checkpoint + `convert_hf_to_gguf.py`.)

### Proxy imatrix / logits

Without `llama-imatrix` / `llama-perplexity`, steps 10–12 use feature/proxy estimates. The recipe is still useful for sizing and planning, but ΔKLD is not measured against real BF16 logits. Install the llama tools and re-run those steps with `--mode llama --force` for measured results.

### Re-run only from export onward

```bash
odg run --model functiongemma:latest --from-step export --no-ask
# or
odg export --model functiongemma:latest --mode llama --force
odg validate --model functiongemma:latest --force
```

---

## Quick checklist — “did I get a real GGUF?”

- [ ] `LLAMA_CPP_DIR` (or `--llama-quantize`) points at a real binary
- [ ] `odg export --mode llama --force` completed without error
- [ ] `steps/14_export/*-UD.gguf` exists (not only `*.gguf.MISSING`)
- [ ] `output.json` has `"method": "llama_quantize"` and non-null `gguf_out`
- [ ] `odg validate` verdict is not stuck on PROVISIONAL for “missing candidate”

---

## 12. Platform commands: fit, benchmark, report

Phase 1 of the [platform expansion](./platform/README.md) adds intent-level commands on top of the pipeline.

### `odg fit` — describe your hardware, not a byte budget

```bash
odg fit --model functiongemma:latest --gpu 12GB
odg fit --model qwen3 --device macbook-air-16gb --ctx 8192
odg fit --model llama4 --ram 32GB --cpu-only
odg devices                       # list named device profiles
```

`fit` resolves the model, derives the weight budget —

```text
budget = memory pool × usable fraction − KV cache(model, ctx) − runtime overhead
```

— prints every subtraction (auditable), saves it as `fit_plan.json` in the run, and then runs the normal pipeline with that budget. The KV cache is exact math from the architecture descriptor (layers × ctx × KV width × dtype).

> Resuming a run that already has an optimized recipe keeps the old recipe; pass `--force` (or `--new-run`) to re-optimize under the new hardware budget.

### `odg benchmark` — comparable numbers for any GGUF

```bash
odg benchmark model-UD.gguf --suite smoke          # minutes
odg benchmark --model functiongemma:latest         # uses the run's exported GGUF
odg benchmark model.gguf --suite standard --device rtx-3060-12gb
```

Writes `benchresult.json` (`odg/benchresult/v1`) with:

- **throughput** via `llama-bench` (needs `LLAMA_CPP_DIR`; skipped honestly otherwise),
- **quality** via lm-eval-harness (`pip install lm-eval` to enable; skipped honestly otherwise) — deltas are paired per-question vs the BF16 reference with bootstrap CIs, never raw thresholds,
- file size + sha256, and the device profile tag.

Results stored under `<run>/benchmarks/` are picked up by the report automatically.

### `odg report` — one self-contained report.html

```bash
odg report --model functiongemma:latest
odg report --model functiongemma:latest --open
```

Renders `report.html` in the run root from artifacts the steps already wrote: bit-allocation table with per-group reasons, byte distribution, sensitivity heatmap, size↔quality Pareto frontier, Tier-1 gates, benchmarks, and a reproducibility block. Sections for steps that haven't run render as "not run". `odg run` / `odg fit` auto-render it after validate.

---

## Related docs

- Project overview: [`../README.md`](../README.md)
- Per-step design docs: [`./steps/README.md`](./steps/README.md)
- Platform feature breakdowns: [`./platform/README.md`](./platform/README.md)
- Export step: [`./steps/14-export-gguf.md`](./steps/14-export-gguf.md)
- Validate step: [`./steps/15-validate-and-release.md`](./steps/15-validate-and-release.md)
