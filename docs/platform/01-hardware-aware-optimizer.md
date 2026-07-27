# Feature 01 — Hardware-aware optimizer

[Index](./README.md) · Next: [02 Benchmark runner](./02-benchmark-runner.md) →

Priority: ⭐⭐⭐⭐⭐ · Phase 1 · New modules: `hardware.py` · New command: `odg fit`

---

## Goal

Users describe their **hardware**, not bits:

```bash
odg fit --model gemma4-27b --gpu 24GB
odg fit --model qwen3 --device macbook-air-16gb --ctx 8192
odg fit --model llama4 --ram 32GB --cpu-only
```

and the platform picks the byte budget, pins, and quant search space for them.

---

## Why it exists

Nobody thinks "3.2 GB target size". They think *"I have a 12 GB GPU"* or *"I have a MacBook
Air"*. Today that translation — VRAM minus runtime overhead minus KV cache at the desired
context length — is done by hand, badly, or not at all. This feature makes it the front door
of the whole tool.

Key insight: **this is a front-end to the existing optimizer, not a new one.** Step 13
already accepts `--budget-mb`. Hardware awareness = profile + context → budget.

---

## Depends on

- Core steps 01–13 (unchanged).
- Nothing else. This can ship first.

---

## Design

### Hardware profile schema (`odg/hardware/v1`)

```yaml
schema: odg/hardware/v1
id: rtx-3060-12gb
kind: gpu                    # gpu | apple_silicon | cpu
vram_gb: 12
ram_gb: 32
bandwidth_gbps: 360
usable_fraction: 0.90        # runtime + fragmentation headroom
kv_cache_reserve_gb: 1.5     # base reserve, scaled by --ctx and model dims
```

### Budget derivation

```text
budget_bytes = vram_gb × usable_fraction
             − kv_cache_bytes(model_dims, ctx, kv_dtype)
             − runtime_overhead(backend)
```

KV cache size is exact math from the architecture descriptor the resolver already produces
(layers × heads × head_dim × ctx × 2 × dtype_bytes). No guessing.

### Profile sources, in priority order

1. `--device <id>` — named profile from a built-in YAML database (`hardware.py` data).
2. `--gpu 24GB` / `--ram 32GB` — ad-hoc profile from flags.
3. Auto-detect — best-effort local probe (`nvidia-smi`, `sysctl hw.memsize`, `/proc/meminfo`)
   with an explicit confirmation line so wrong detection never silently misallocates.

### CLI

`odg fit` = derive budget → print the plan → run the standard pipeline with that budget:

```text
$ odg fit --model gemma4-27b --gpu 12GB --ctx 8192

  Profile      : gpu 12.0 GB (ad-hoc)
  Usable       : 10.8 GB  (0.90 headroom)
  KV cache     : 1.9 GB   (8192 ctx, f16)
  Weight budget: 8.9 GB   → passed to optimizer as --budget-mb 9113

  Proceeding: resolve → … → optimize → export → validate
```

---

## Build steps

Each step is a small PR, independently mergeable.

1. **`hardware.py`: profile dataclass + schema validation.** Load/validate
   `odg/hardware/v1` YAML. Unit tests only, no CLI yet.
2. **Built-in device database.** Start with ~15 common profiles: RTX 3060/3090/4090,
   RX 7900, M-series Macs at 8/16/32/64 GB, generic `cpu-<N>gb`. One YAML file, easy PRs.
3. **KV-cache calculator.** Exact bytes from the architecture descriptor (step 01 output) +
   ctx + kv dtype. Cross-check against llama.cpp's reported allocation on one known model.
4. **Budget derivation function.** profile + descriptor + ctx → `--budget-mb`. Pure
   function, table-driven tests.
5. **`odg fit` command.** Wire flags (`--gpu/--ram/--device/--ctx/--cpu-only`), print the
   plan block, delegate to the existing run driver with the derived budget.
6. **Auto-detection (optional last).** Local probe with confirmation prompt; `--yes` for
   scripts.
7. **Docs + examples.** Update `USAGE.md`; add `odg fit` to README quick start.

---

## Done when

- [ ] `odg fit --model X --gpu 12GB` runs the full pipeline with a correct derived budget
- [ ] KV-cache math validated against a real llama.cpp run (within 5%)
- [ ] Named profiles resolve (`--device rtx-3060`), unknown ids fail with the list
- [ ] The printed plan shows every subtraction, so the budget is auditable
- [ ] Ad-hoc, named, and detected profiles all produce an `odg/hardware/v1` artifact stored
      in the run (needed later by reports and model cards)

## Next

[Feature 02 — Benchmark runner](./02-benchmark-runner.md)
