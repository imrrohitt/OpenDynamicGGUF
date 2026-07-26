# Step 14 — Export the quantized GGUF from a recipe

← [13 Optimize](./13-optimize-recipe.md) · [Index](./README.md) · Next: [15 Validate](./15-validate-and-release.md) →

---

## Goal

Render the recipe into `llama-quantize` and produce a candidate `*-UD.gguf` (or a dry-run command when tools are missing).

---

## Command

```bash
odg export --model functiongemma:latest

# require real quantize:
export LLAMA_CPP_DIR=~/llama.cpp
odg export --model functiongemma:latest --mode llama --force
```

Requires Step 13.

---

## Outputs

```text
steps/14_export/
  recipe.yaml / recipe.tt     # copied for provenance
  <model>-UD.gguf             # when llama-quantize succeeds
  *.gguf.MISSING              # dry-run marker
  quantize_command.sh
  export_manifest.json
  output.json
```

---

## Done when

- [x] Quantize command recorded with recipe + GGUF hashes
- [x] Real GGUF **or** documented dry-run

## Next

[Step 15 — Validate and release](./15-validate-and-release.md)
