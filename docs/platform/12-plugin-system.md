# Feature 12 — Plugin system

← [11 Automatic model cards](./11-automatic-model-cards.md) · [Index](./README.md) · Next: [13 Security scanner](./13-security-scanner.md) →

Priority: ⭐⭐⭐⭐☆ · Phase 4 (interfaces introduced earlier, in features 02/08/09) · New modules: `plugins.py`

---

## Goal

Third parties extend the platform without touching core:

```toml
# a plugin package's pyproject.toml
[project.entry-points."odg.metrics"]
js_divergence = "odg_jsd.metric:JSDivergence"

[project.entry-points."odg.search"]
anneal = "odg_anneal.search:SimulatedAnnealing"

[project.entry-points."odg.exporters"]
mlx = "odg_mlx.export:MlxExporter"

[project.entry-points."odg.evals"]
sql_suite = "odg_sql.evals:SqlEvalSuite"
```

```bash
pip install odg-anneal
odg optimize --model x --search anneal      # just works
odg plugins list
```

---

## Why it exists

Every "can you add metric/algorithm/format X" issue currently means a core PR. Four seams
already exist implicitly — metrics (steps 12/15), search (step 13 / feature 08), exporters
(step 14), eval suites (feature 02) — this feature makes them official so the community can
build on the platform instead of forking it (platform invariant 6).

---

## Depends on

- Feature 08 defines the search interface, feature 09 the metric interface, feature 02 the
  eval-suite interface. This feature unifies them under discovery + registration.

---

## Design

### Four extension points, four small ABCs

| Entry point group | ABC | Contract |
|---|---|---|
| `odg.metrics` | `Metric` | `score(candidate_artifacts, context) → MetricResult`; declares cost tier (free / cheap / measured) |
| `odg.search` | `SearchAlgorithm` | `run(space, evaluate, budget) → frontier` (feature 08's interface) |
| `odg.exporters` | `Exporter` | `export(recipe, source_ref, outdir) → artifact + hash` |
| `odg.evals` | `EvalSuite` | task list + config → `odg/benchresult/v1` fragment |

### Rules

1. **Built-ins are plugins.** KLD metric, greedy search, llama.cpp exporter, and the
   standard suites register through the same entry points — the plugin path is exercised on
   every run, so it cannot rot.
2. **Version handshake.** Each ABC has an API version; plugins declare compatibility, loader
   refuses mismatches with a clear message instead of a stack trace mid-run.
3. **Provenance taint.** Artifacts produced with third-party plugins record the plugin name +
   version + package hash. Marketplace/leaderboard submissions using non-built-in metrics are
   flagged as such (the trust chain must stay auditable).
4. **Failure isolation.** A plugin exception fails its step with attribution ("plugin
   odg-anneal 0.2.1 raised …"), never corrupts the run store.

---

## Build steps

1. **`plugins.py` loader.** `importlib.metadata` entry-point discovery, API-version check,
   registry with name collision detection. `odg plugins list`.
2. **Extract `Metric` ABC.** Move the existing KLD scoring behind it; step 12/15 consume via
   the registry. No behavior change — the regression test is "identical artifacts".
3. **Extract `SearchAlgorithm`.** Greedy (step 13) becomes the built-in registered
   implementation (aligns with feature 08 step 1 if that lands first).
4. **Extract `Exporter`.** llama.cpp export path (step 14) behind the ABC.
5. **`EvalSuite`.** Feature 02 suites move to entry points.
6. **Provenance taint + docs.** Plugin identity in artifact metadata; publishing rules wired
   into features 04/05.
7. **Example plugin repo.** A real, minimal external package (e.g. a toy metric) serving as
   the template + integration test.

---

## Done when

- [ ] All four built-in implementations load through entry points (no special-case paths)
- [ ] A pip-installed external plugin is usable by name with zero core changes
- [ ] API-version mismatch produces a clear refusal, not a crash
- [ ] Plugin-produced artifacts are taint-labeled through to marketplace/leaderboard
- [ ] Example plugin repo published and linked from contributing docs

## Next

[Feature 13 — Security scanner](./13-security-scanner.md)
