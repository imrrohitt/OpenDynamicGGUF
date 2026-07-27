# Feature 15 — Web UI

← [14 CI/CD GitHub Action](./14-cicd-github-action.md) · [Index](./README.md)

Priority: ⭐⭐⭐⭐☆ · Phase 5 · New tree: `webui/` · New command: `odg ui`

---

## Goal

A browser interface for non-CLI users:

```bash
odg ui        # → http://localhost:8000
```

```text
Pick model (HF id / local path / Ollama tag)
  ↓
Select hardware (device picker or auto-detect)
  ↓
Select workload + objective + target size
  ↓
Optimize (live step progress, resumable)
  ↓
Explore report → Download GGUF + recipe
```

---

## Why it exists

Every earlier feature made decisions easier for CLI users; this one removes the CLI
requirement entirely, broadening the audience to anyone who can run one command and open a
browser. It ships last on purpose: it is a **thin client over a finished engine** — the run
store, `odg fit`, reports, and explanations all exist by now, so the UI invents nothing.

---

## Depends on

- Features 01 (hardware picker), 03 (report sections reused as UI views), 10 (explanations),
  04 (recipe browser), optionally 06 (publish button).
- Platform invariant 1: no optimization logic in the UI.

---

## Design

### Architecture

```text
browser (static SPA)  ⇄  FastAPI server (webui/)  ⇄  run store + step driver
```

- **Local-first.** `odg ui` serves on localhost; the model, the compute, and the artifacts
  are the user's own machine. No hosted service in scope for v1.
- **The API mirrors the CLI 1:1.** `POST /runs` = `odg fit`; `GET /runs/{id}` = `odg status`;
  every endpoint is a thin wrapper over the same functions `cli.py` calls. If an endpoint
  needs new logic, that logic goes into core first.
- **Progress = the run store.** The 15-step checkpoint layout already is a progress model;
  the UI renders `status.json` per step, streams `log.txt` tails over SSE, and resumes
  interrupted runs exactly like the CLI does.
- **Report views reused.** The report's chart components (feature 03) render live inside the
  UI instead of being reimplemented.

### Screens

| Screen | Backed by |
|---|---|
| New run wizard (model → hardware → workload → budget) | resolver check, hardware DB, corpus presets, budget preview from feature 01 |
| Run monitor (step list, live logs, cancel/resume) | run store + SSE |
| Results (allocations, explanations, KLD, Pareto picker) | report data + `explain.json` |
| Recipe browser (marketplace search, one-click build) | `registry.py` |
| Downloads (GGUF, recipe, report, card) | run artifacts |

---

## Build steps

1. **API server.** FastAPI over the run store: list runs, step status, log tail (SSE),
   start/resume/cancel runs. No frontend yet — usable via curl.
2. **Run monitor page.** The read-only half first: watch an existing CLI-started run in the
   browser. Immediately useful, zero risk.
3. **New-run wizard.** Hardware picker (device DB + detect), workload picker, live budget
   preview ("your 12 GB → 8.9 GB weight budget").
4. **Results screen.** Embed report sections + explanation table; Pareto point picker
   triggers export of the chosen alternative.
5. **Recipe browser.** Marketplace list/search → prefill the wizard with a recipe build.
6. **Packaging.** Frontend built + bundled into the wheel so `pip install` + `odg ui` works
   offline with no Node toolchain.

---

## Done when

- [ ] `odg ui` gives model → hardware → optimized GGUF download without touching a terminal
      again
- [ ] UI-started runs are plain run-store runs — inspectable and resumable from the CLI
- [ ] Live step progress + logs stream during a run; interrupted runs resume from the UI
- [ ] Recipe browser builds a marketplace recipe end-to-end
- [ ] Zero optimization logic in `webui/` (review checklist item)

---

*End of feature breakdowns — back to the [index](./README.md) or the
[platform architecture](../../ARCHITECTURE.md).*
