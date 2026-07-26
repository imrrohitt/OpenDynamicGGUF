# Checkpoint store (filesystem)

← [Index](./README.md)

---

## Goal

Never lose progress mid-pipeline. Every step writes durable files under `artifacts/runs/<run_id>/`. If a step crashes, you can resume from the last **done** checkpoint.

---

## Layout

```text
artifacts/
├── runs/
│   └── 20260726-204800-functiongemma-latest/
│       ├── run.json                 # run metadata + all step statuses
│       └── steps/
│           ├── 01_resolve/
│           │   ├── status.json      # pending | running | done | failed
│           │   ├── input.json       # what you asked for
│           │   ├── output.json      # step result (ResolvedModel, …)
│           │   ├── log.txt          # human step log
│           │   └── error.txt        # only if failed
│           ├── 02_load/
│           ├── …
│           └── 15_validate/
└── models/
    └── functiongemma-latest/
        └── CURRENT                  # active run_id for this model
```

---

## Commands

```bash
# Step 01 — creates/resumes a run and checkpoints resolve
odg resolve --model functiongemma:latest

# Show board of all 15 steps for the current run
odg status --model functiongemma:latest

# List runs
odg runs

# Force re-run step 01 (overwrite checkpoint)
odg resolve --model functiongemma:latest --force

# Brand-new run (do not resume CURRENT)
odg resolve --model functiongemma:latest --new-run
```

---

## How later steps should use it

```python
from store import RunStore, StepAlreadyDone

store = RunStore()  # ./artifacts
meta = store.get_or_create_run("functiongemma:latest")

if store.is_step_done(meta.run_id, "resolve"):
    resolved = store.read_step_output(meta.run_id, "resolve")

store.begin_step(meta.run_id, "catalog", input_data={...})
# ... work ...
store.complete_step(meta.run_id, "catalog", output_data={...}, log_text="...")
# on error:
# store.fail_step(meta.run_id, "catalog", str(exc))
```

---

## Why this matters

Without a store, a crash at Step 12 means re-doing catalog, corpus, imatrix, and probes. With checkpoints, Step 12 resumes from saved sensitivity partials / previous done steps.

Code: `store.py`, `steps.py`.
