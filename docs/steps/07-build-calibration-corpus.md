# Step 07 — Build the calibration corpus (3-way split)

← [06 Weight features](./06-compute-weight-features.md) · [Index](./README.md) · Next: [08 Activation features](./08-compute-activation-features.md) →

---

## Goal

Create the **text** the model will be run on for activation stats, imatrix, KL search, and final validation — and split it so search never sees the judging set.

---

## Why it exists

Activation stats, imatrix, and ΔKLD all need **real prompts**:

```text
Prompt → Model → Activations / logits → Metrics
```

Wikipedia-only calibration overfits Wikipedia-style KL scores. Instruct models also need **chat-template** formatting.

---

## Inputs / Outputs

| | |
|---|---|
| **Input** | Architecture descriptor (chat template, specialty domain) |
| **Output** | `calib.txt`, `search.txt`, `heldout.txt` (+ manifest) |

---

## How it works

### Mixed domain mix (default)

| Domain | Share | Examples |
|---|---|---|
| Conversation | ~30% | “Explain quantum mechanics…” |
| Code | ~30% | “Write a Python merge of two sorted lists…” |
| Math / reasoning | ~20% | “Solve 25 × 37…” |
| Multilingual | ~10% | Translate / non-English chat |
| Domain-specific | ~10% | From resolver (e.g. tool-call traces) |

### FunctionGemma example (domain-specific)

Include real function-calling traces rendered with its template:

```text
<start_of_turn>user
What's the weather in Paris?
<end_of_turn>
<start_of_turn>model
call weather_api(city="Paris")
<end_of_turn>
```

### Three-way split

```text
Full corpus (~0.3M–1.5M tokens)
        │
        ├── 60%  calib.txt     → imatrix + activation features
        ├── 20%  search.txt    → ΔKLD during probing / optimize
        └── 20%  heldout.txt   → validation ONLY (never for search)
```

Hard rule: **optimizer must not read held-out.**

### Build sketch

```python
def build_corpus(descriptor) -> CorpusPaths:
    chunks = []
    chunks += load_chat(share=0.30)
    chunks += load_code(share=0.30)
    chunks += load_math(share=0.20)
    chunks += load_multi(share=0.10)
    chunks += load_domain(descriptor.specialty_domain, share=0.10)

    rendered = [apply_chat_template(c, descriptor.chat_template) for c in chunks]
    calib, search, heldout = split(rendered, 0.6, 0.2, 0.2, seed=42)
    write_text("calib.txt", calib)
    write_text("search.txt", search)
    write_text("heldout.txt", heldout)
    return CorpusPaths(...)
```

---

## Example snippet (`calib.txt`)

```text
User: Explain quantum mechanics in simple terms.
Assistant: Quantum mechanics describes nature at very small scales...

User: Write Python code to reverse a linked list.
Assistant: ```python
def reverse(head):
    prev = None
    ...
```

User: Solve: 25 × 37
Assistant: 925
```

---

## Done when

- [ ] Three files written with disjoint content
- [ ] Chat template applied for instruct models
- [ ] Domain data included when specialty_domain is set
- [ ] Token counts logged in a corpus manifest

## Next

[Step 08 — Compute activation features](./08-compute-activation-features.md)
