"""Step 07 — calibration corpus (3-way split)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import hashlib
import random
from pathlib import Path


# --- from corpus/types.py ---
@dataclass
class CorpusResult:
    model_ref: str
    corpus_id: str
    chat_template: str | None
    specialty_domain: str | None
    seed: int
    target_tokens: int
    splits: dict[str, float]
    n_documents: int
    n_calib: int
    n_search: int
    n_heldout: int
    tokens_est_total: int
    tokens_est_calib: int
    tokens_est_search: int
    tokens_est_heldout: int
    chars_total: int
    domain_counts: dict[str, int]
    files: dict[str, str]
    steps_log: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary_dict(self) -> dict[str, Any]:
        return asdict(self)

# --- from corpus/templates.py ---
def render_chat(
    user: str,
    assistant: str | None = None,
    *,
    template: str | None = "gemma3",
) -> str:
    """Render a user(/assistant) turn with the model's chat template."""
    t = (template or "plain").lower()
    if t in {"gemma", "gemma2", "gemma3", "functiongemma"}:
        return _gemma(user, assistant)
    if t in {"chatml", "qwen", "qwen2"}:
        return _chatml(user, assistant)
    if t in {"llama3", "llama-3"}:
        return _llama3(user, assistant)
    return _plain(user, assistant)


def render_document(doc: dict[str, Any], *, template: str | None) -> str:
    """
    doc keys:
      user / assistant  — single turn
      turns             — list of {role, content}
      text              — already-formatted raw text (no template)
    """
    if "text" in doc and doc["text"]:
        return str(doc["text"]).strip() + "\n"

    turns = doc.get("turns")
    if turns:
        return render_turns(turns, template=template)

    user = str(doc.get("user") or "").strip()
    assistant = doc.get("assistant")
    assistant_s = str(assistant).strip() if assistant is not None else None
    return render_chat(user, assistant_s, template=template)


def render_turns(turns: list[dict[str, str]], *, template: str | None) -> str:
    t = (template or "plain").lower()
    if t in {"gemma", "gemma2", "gemma3", "functiongemma"}:
        parts: list[str] = []
        for turn in turns:
            role = turn.get("role", "user")
            content = (turn.get("content") or "").strip()
            if role == "system":
                # Gemma has no system turn; prepend to first user
                continue
            gemma_role = "model" if role in {"assistant", "model"} else "user"
            parts.append(f"<start_of_turn>{gemma_role}\n{content}<end_of_turn>")
        # If system present, prepend to first user content
        sys_msgs = [x.get("content", "") for x in turns if x.get("role") == "system"]
        if sys_msgs and parts:
            sys_text = "\n".join(s.strip() for s in sys_msgs if s).strip()
            if sys_text and parts[0].startswith("<start_of_turn>user\n"):
                parts[0] = parts[0].replace(
                    "<start_of_turn>user\n",
                    f"<start_of_turn>user\n{sys_text}\n\n",
                    1,
                )
        return "\n".join(parts) + "\n"

    # Fallback: concatenate plain turns
    lines = []
    for turn in turns:
        role = turn.get("role", "user").capitalize()
        lines.append(f"{role}: {(turn.get('content') or '').strip()}")
    return "\n".join(lines) + "\n"


def _gemma(user: str, assistant: str | None) -> str:
    parts = [f"<start_of_turn>user\n{user.strip()}<end_of_turn>"]
    if assistant:
        parts.append(f"<start_of_turn>model\n{assistant.strip()}<end_of_turn>")
    return "\n".join(parts) + "\n"


def _chatml(user: str, assistant: str | None) -> str:
    parts = [
        "<|im_start|>user\n" + user.strip() + "<|im_end|>",
    ]
    if assistant:
        parts.append("<|im_start|>assistant\n" + assistant.strip() + "<|im_end|>")
    return "\n".join(parts) + "\n"


def _llama3(user: str, assistant: str | None) -> str:
    parts = [
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        + user.strip()
        + "<|eot_id|>"
    ]
    if assistant:
        parts.append(
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            + assistant.strip()
            + "<|eot_id|>"
        )
    return "".join(parts) + "\n"


def _plain(user: str, assistant: str | None) -> str:
    if assistant:
        return f"User: {user.strip()}\nAssistant: {assistant.strip()}\n"
    return f"User: {user.strip()}\n"

# --- from corpus/prompts.py ---
def conversation_bank() -> list[dict[str, Any]]:
    topics = [
        ("quantum mechanics", "It describes nature at atomic scales using probabilities and wavefunctions."),
        ("photosynthesis", "Plants convert light, water, and CO2 into sugar and oxygen via chlorophyll."),
        ("the water cycle", "Water evaporates, condenses into clouds, then returns as precipitation."),
        ("machine learning", "Algorithms learn patterns from data to make predictions without hard-coded rules."),
        ("black holes", "Regions where gravity is so strong that nothing, not even light, can escape."),
        ("inflation", "A sustained rise in the general price level that reduces purchasing power."),
        ("DNA", "A double helix of nucleotides that stores genetic instructions for living organisms."),
        ("democracy", "A system where citizens choose leaders and influence laws through voting."),
        ("relativity", "Einstein's theory relating space, time, and gravity; mass and energy are equivalent."),
        ("climate change", "Long-term shifts in temperature and weather patterns, largely from greenhouse gases."),
        ("antibiotics", "Drugs that kill or inhibit bacteria; overuse can lead to resistance."),
        ("the Internet", "A global network of computers communicating via standardized protocols."),
        ("supply and demand", "Prices tend to rise when demand exceeds supply and fall in the opposite case."),
        ("vaccines", "Preparations that train the immune system to recognize pathogens safely."),
        ("plate tectonics", "Earth's crust is divided into plates that move, causing quakes and volcanoes."),
        ("blockchain", "A distributed ledger where blocks of transactions are cryptographically linked."),
        ("memory in the brain", "Encoding, storage, and retrieval of information across neural networks."),
        ("renewable energy", "Power from sources that replenish naturally, like solar, wind, and hydro."),
        ("evolution", "Populations change over generations via mutation, selection, and drift."),
        ("cryptography", "Techniques for secure communication in the presence of adversaries."),
    ]
    out = []
    for topic, answer in topics:
        out.append(
            {
                "domain": "conversation",
                "user": f"Explain {topic} in simple terms.",
                "assistant": answer,
            }
        )
        out.append(
            {
                "domain": "conversation",
                "user": f"Give three practical examples related to {topic}.",
                "assistant": f"Here are three concrete examples involving {topic}, each illustrating a different everyday use-case.",
            }
        )
    extras = [
        "Summarize the plot of a mystery novel in two paragraphs without spoilers for the ending.",
        "What habits help someone learn a new language faster?",
        "Compare tea and coffee in terms of caffeine, taste, and cultural roles.",
        "How should a beginner start investing with a small monthly budget?",
        "Describe a productive morning routine for a remote software engineer.",
        "What is the difference between weather and climate?",
        "Explain empathy and why it matters in teamwork.",
        "List pros and cons of living in a large city versus a small town.",
    ]
    for u in extras:
        out.append({"domain": "conversation", "user": u, "assistant": None})
    return out


def code_bank() -> list[dict[str, Any]]:
    items = [
        (
            "Write a Python function that merges two sorted lists into one sorted list.",
            "def merge(a, b):\n    i = j = 0\n    out = []\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n            out.append(b[j]); j += 1\n    out.extend(a[i:])\n    out.extend(b[j:])\n    return out",
        ),
        (
            "Write a Python function to reverse a singly linked list.",
            "def reverse(head):\n    prev = None\n    cur = head\n    while cur:\n        nxt = cur.next\n        cur.next = prev\n        prev = cur\n        cur = nxt\n    return prev",
        ),
        (
            "Write a TypeScript function that debounces another function by delay ms.",
            "export function debounce<T extends (...args: any[]) => void>(fn: T, delay: number) {\n  let t: ReturnType<typeof setTimeout> | undefined;\n  return (...args: Parameters<T>) => {\n    clearTimeout(t);\n    t = setTimeout(() => fn(...args), delay);\n  };\n}",
        ),
        (
            "Write a SQL query to find the top 5 customers by total order amount.",
            "SELECT customer_id, SUM(amount) AS total\nFROM orders\nGROUP BY customer_id\nORDER BY total DESC\nLIMIT 5;",
        ),
        (
            "Write a Bash one-liner to count unique lines in a file.",
            "sort file.txt | uniq | wc -l",
        ),
        (
            "Implement binary search in Python that returns the index or -1.",
            "def binary_search(arr, x):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == x:\n            return mid\n        if arr[mid] < x:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
        ),
        (
            "Write a Rust function that returns the nth Fibonacci number iteratively.",
            "fn fib(n: u32) -> u64 {\n    let (mut a, mut b) = (0u64, 1u64);\n    for _ in 0..n { let t = a + b; a = b; b = t; }\n    a\n}",
        ),
        (
            "Write a Go HTTP handler that returns JSON {\"ok\": true}.",
            "func ok(w http.ResponseWriter, r *http.Request) {\n    w.Header().Set(\"Content-Type\", \"application/json\")\n    w.Write([]byte(`{\"ok\":true}`))\n}",
        ),
        (
            "Explain how to fix a Python ImportError for a local package.",
            "Ensure the package directory has __init__.py, install it editable with pip install -e ., or add the project root to PYTHONPATH.",
        ),
        (
            "Write a regex to validate a simple email address and explain limitations.",
            r"Pattern: ^[^@\s]+@[^@\s]+\.[^@\s]+$ — accepts many valid forms but not full RFC compliance.",
        ),
        (
            "Write a Python generator that yields sliding windows of size k over a list.",
            "def windows(xs, k):\n    for i in range(len(xs) - k + 1):\n        yield xs[i:i+k]",
        ),
        (
            "Show a minimal Dockerfile for a FastAPI app on port 8000.",
            "FROM python:3.12-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install -r requirements.txt\nCOPY . .\nCMD [\"uvicorn\", \"main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]",
        ),
    ]
    out = [{"domain": "code", "user": u, "assistant": a} for u, a in items]
    more = [
        "Refactor this idea: a function that retries an HTTP GET up to 3 times with backoff.",
        "What is the difference between threads and asyncio in Python?",
        "Write unit tests for a function that clamps a number into [lo, hi].",
        "Explain Big-O of inserting into a balanced BST versus a hash map.",
        "Convert a nested JSON object into a flat key.path map in Python.",
    ]
    for u in more:
        out.append({"domain": "code", "user": u, "assistant": None})
    return out


def math_bank() -> list[dict[str, Any]]:
    items = [
        ("Solve: 25 × 37. Show steps.", "25 × 37 = 25 × (30 + 7) = 750 + 175 = 925."),
        ("What is 15% of 240?", "0.15 × 240 = 36."),
        ("Simplify: (3/4) + (5/6).", "LCD 12 → 9/12 + 10/12 = 19/12."),
        ("If a train travels 90 km in 1.5 hours, what is its average speed?", "90 / 1.5 = 60 km/h."),
        ("Solve for x: 2x + 7 = 19.", "2x = 12 → x = 6."),
        ("Compute the area of a circle with radius 5 (use π≈3.1416).", "πr² ≈ 3.1416 × 25 = 78.54."),
        ("A shirt costs $40 after a 20% discount. What was the original price?", "40 = 0.8P → P = 50."),
        ("How many ways can you arrange 5 distinct books on a shelf?", "5! = 120."),
        ("Convert 72°F to Celsius.", "C = (72 − 32) × 5/9 = 40 × 5/9 ≈ 22.22°C."),
        ("Find the mean of 4, 8, 15, 16, 23, 42.", "(4+8+15+16+23+42)/6 = 108/6 = 18."),
        ("Is 97 prime? Explain briefly.", "Yes — no divisors among primes ≤ √97 ≈ 9.8 (2,3,5,7)."),
        ("Solve the system: x+y=10, x−y=2.", "Add: 2x=12 → x=6; y=4."),
    ]
    out = [{"domain": "math", "user": u, "assistant": a} for u, a in items]
    for n in range(2, 30):
        out.append(
            {
                "domain": "math",
                "user": f"Compute {n}! step by step and give the final integer.",
                "assistant": None,
            }
        )
    return out


def multilingual_bank() -> list[dict[str, Any]]:
    items = [
        ("Translate to French: Good morning, how are you?", "Bonjour, comment allez-vous ?"),
        ("Traduce al español: The library opens at nine.", "La biblioteca abre a las nueve."),
        ("Übersetze ins Deutsche: I need a ticket to Berlin.", "Ich brauche eine Fahrkarte nach Berlin."),
        ("हिंदी में अनुवाद करें: Please wait a moment.", "कृपया एक क्षण प्रतीक्षा करें।"),
        ("Traduisez en anglais: Je voudrais un café, s'il vous plaît.", "I would like a coffee, please."),
        ("Reply in Japanese: Thank you for your help.", "助けてくれてありがとう。"),
        ("Translate to Italian: Where is the train station?", "Dov'è la stazione dei treni?"),
        ("Responda em português: What time is dinner?", "A que horas é o jantar?"),
        ("Explain in simple English, then give a Hindi summary: what is gravity?", None),
        ("Write a short polite email in Spanish asking to reschedule a meeting.", None),
    ]
    return [
        {"domain": "multilingual", "user": u, "assistant": a} for u, a in items
    ]


def function_calling_bank() -> list[dict[str, Any]]:
    """FunctionGemma-style tool traces (domain specialty)."""
    traces = [
        {
            "domain": "function_calling",
            "turns": [
                {"role": "user", "content": "What's the weather in Paris?"},
                {
                    "role": "model",
                    "content": 'call weather_api(city="Paris")',
                },
            ],
        },
        {
            "domain": "function_calling",
            "turns": [
                {"role": "user", "content": "Set a reminder for tomorrow at 9am to call Mom."},
                {
                    "role": "model",
                    "content": 'call create_reminder(title="Call Mom", time="tomorrow 09:00")',
                },
            ],
        },
        {
            "domain": "function_calling",
            "turns": [
                {"role": "user", "content": "Search my calendar for meetings next Monday."},
                {
                    "role": "model",
                    "content": 'call calendar_search(query="meetings", date="next Monday")',
                },
            ],
        },
        {
            "domain": "function_calling",
            "turns": [
                {"role": "user", "content": "Convert 100 USD to EUR."},
                {
                    "role": "model",
                    "content": 'call currency_convert(amount=100, from="USD", to="EUR")',
                },
            ],
        },
        {
            "domain": "function_calling",
            "turns": [
                {"role": "user", "content": "Book a table for 2 at Luigi's tonight at 7."},
                {
                    "role": "model",
                    "content": 'call restaurant_book(name="Luigi\'s", party_size=2, time="19:00")',
                },
            ],
        },
        {
            "domain": "function_calling",
            "turns": [
                {"role": "user", "content": "Send an email to alex@example.com saying I'll be late."},
                {
                    "role": "model",
                    "content": 'call send_email(to="alex@example.com", subject="Running late", body="I\'ll be late.")',
                },
            ],
        },
        {
            "domain": "function_calling",
            "turns": [
                {"role": "user", "content": "Turn on the living room lights to 40%."},
                {
                    "role": "model",
                    "content": 'call smart_home(device="living_room_lights", action="set_brightness", value=40)',
                },
            ],
        },
        {
            "domain": "function_calling",
            "turns": [
                {"role": "user", "content": "Find flights from SFO to JFK on August 12."},
                {
                    "role": "model",
                    "content": 'call flight_search(origin="SFO", destination="JFK", date="2026-08-12")',
                },
            ],
        },
        {
            "domain": "function_calling",
            "user": "You have tools: weather_api, calendar_search. User asks: Will it rain during my 3pm meeting in Seattle?",
            "assistant": 'call weather_api(city="Seattle")\ncall calendar_search(query="meeting", time="15:00")',
        },
        {
            "domain": "function_calling",
            "user": "List the JSON schema fields for a tool named get_stock_price(symbol).",
            "assistant": '{"name":"get_stock_price","parameters":{"type":"object","properties":{"symbol":{"type":"string"}},"required":["symbol"]}}',
        },
    ]
    # Expand with city variants
    cities = ["Tokyo", "London", "Mumbai", "São Paulo", "Cairo", "Sydney", "Toronto"]
    for city in cities:
        traces.append(
            {
                "domain": "function_calling",
                "turns": [
                    {"role": "user", "content": f"What's the weather in {city}?"},
                    {
                        "role": "model",
                        "content": f'call weather_api(city="{city}")',
                    },
                ],
            }
        )
    return traces


def domain_bank(specialty: str | None) -> list[dict[str, Any]]:
    if specialty in {"function_calling", "tools", "tool_use"}:
        return function_calling_bank()
    if specialty in {"code", "coding"}:
        return code_bank()[:8]
    return []


SHARES = {
    "conversation": 0.30,
    "code": 0.30,
    "math": 0.20,
    "multilingual": 0.10,
    "domain": 0.10,
}

# --- from corpus/build.py ---
CORPUS_ID = "odg-corpus-v1"
DEFAULT_SPLITS = {"calib": 0.6, "search": 0.2, "heldout": 0.2}


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars / token). Good enough for manifests."""
    return max(1, (len(text) + 3) // 4) if text else 0


def _vary(doc: dict[str, Any], variant: int) -> dict[str, Any]:
    """Light variation so tiled corpus isn't byte-identical."""
    d = dict(doc)
    tag = f" [ex{variant}]"
    if "user" in d and d["user"]:
        d["user"] = str(d["user"]) + tag
    if "turns" in d and d["turns"]:
        turns = [dict(t) for t in d["turns"]]
        for t in turns:
            if t.get("role") in {"user", "User"}:
                t["content"] = str(t.get("content") or "") + tag
                break
        d["turns"] = turns
    if "text" in d and d["text"]:
        d["text"] = str(d["text"]) + tag
    return d


def _collect_banks(specialty: str | None) -> dict[str, list[dict[str, Any]]]:
    return {
        "conversation": conversation_bank(),
        "code": code_bank(),
        "math": math_bank(),
        "multilingual": multilingual_bank(),
        "domain": domain_bank(specialty),
    }


def build_document_pool(
    *,
    specialty_domain: str | None,
    chat_template: str | None,
    target_tokens: int,
    seed: int,
) -> tuple[list[str], dict[str, int], list[str]]:
    """
    Mix domains by share, render with chat template, tile until ~target_tokens.
    Returns (documents, domain_counts, log).
    """
    log: list[str] = []
    rng = random.Random(seed)
    banks = _collect_banks(specialty_domain)

    # If no specialty domain bank, redistribute that share to conversation/code
    shares = dict(SHARES)
    if not banks["domain"]:
        shares["conversation"] += shares["domain"] / 2
        shares["code"] += shares["domain"] / 2
        shares["domain"] = 0.0
        log.append("1. No specialty domain bank — redistributed domain share")
    else:
        log.append(
            f"1. Specialty domain={specialty_domain!r} "
            f"({len(banks['domain'])} seed prompts)"
        )

    # Build a mixed ordered pool (one pass over banks, shuffled)
    mixed: list[tuple[str, dict[str, Any]]] = []
    for domain, share in shares.items():
        if share <= 0:
            continue
        bank = list(banks[domain])
        if not bank:
            continue
        # Approximate count by share relative to conversation size as baseline
        # We'll tile later; here just ensure each domain appears.
        rng.shuffle(bank)
        for doc in bank:
            mixed.append((domain, doc))

    rng.shuffle(mixed)
    log.append(
        f"2. Seed pool size={len(mixed)} domains="
        + ",".join(f"{k}:{len(banks[k])}" for k in banks)
    )
    log.append(f"3. Chat template={chat_template!r} target_tokens≈{target_tokens}")

    docs: list[str] = []
    domain_counts: dict[str, int] = {k: 0 for k in shares}
    tokens = 0
    variant = 0

    # Cycle through mixed pool with variations until token budget
    while tokens < target_tokens:
        variant += 1
        order = list(mixed)
        rng.shuffle(order)
        for domain, doc in order:
            rendered = render_document(_vary(doc, variant), template=chat_template)
            docs.append(rendered)
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            tokens += estimate_tokens(rendered)
            if tokens >= target_tokens:
                break
        if not mixed:
            break

    log.append(
        f"4. Rendered documents={len(docs)} tokens_est={tokens} "
        f"chars={sum(len(d) for d in docs)}"
    )
    return docs, domain_counts, log


def split_documents(
    docs: list[str],
    *,
    splits: dict[str, float] | None = None,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Deterministic disjoint 3-way split."""
    splits = splits or dict(DEFAULT_SPLITS)
    rng = random.Random(seed)
    idxs = list(range(len(docs)))
    rng.shuffle(idxs)

    n = len(idxs)
    n_calib = int(n * splits["calib"])
    n_search = int(n * splits["search"])
    # remainder → heldout (ensures all docs used)
    calib_i = idxs[:n_calib]
    search_i = idxs[n_calib : n_calib + n_search]
    held_i = idxs[n_calib + n_search :]

    return {
        "calib": [docs[i] for i in calib_i],
        "search": [docs[i] for i in search_i],
        "heldout": [docs[i] for i in held_i],
    }


def write_split_file(path: Path, documents: list[str]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Separate docs with blank line for readability; llama-imatrix treats as text
    body = "\n".join(d.rstrip() + "\n" for d in documents)
    path.write_text(body, encoding="utf-8")
    return {
        "path": str(path),
        "n_documents": len(documents),
        "chars": len(body),
        "tokens_est": estimate_tokens(body),
        "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
    }


def build_corpus(
    *,
    model_ref: str,
    out_dir: Path,
    chat_template: str | None = "gemma3",
    specialty_domain: str | None = None,
    target_tokens: int = 50_000,
    seed: int = 42,
    splits: dict[str, float] | None = None,
) -> tuple[CorpusResult, dict[str, Any]]:
    """
    Build corpus into out_dir.

    Writes: calib.txt, search.txt, heldout.txt, corpus_manifest.json
    """
    splits = splits or dict(DEFAULT_SPLITS)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    docs, domain_counts, log = build_document_pool(
        specialty_domain=specialty_domain,
        chat_template=chat_template,
        target_tokens=target_tokens,
        seed=seed,
    )
    parts = split_documents(docs, splits=splits, seed=seed)
    log.append(
        f"5. Split seed={seed} calib={len(parts['calib'])} "
        f"search={len(parts['search'])} heldout={len(parts['heldout'])}"
    )

    meta_files = {}
    for name in ("calib", "search", "heldout"):
        meta_files[name] = write_split_file(out_dir / f"{name}.txt", parts[name])

    # Disjointness check
    sets = {k: set(v) for k, v in parts.items()}
    overlap_cs = len(sets["calib"] & sets["search"])
    overlap_ch = len(sets["calib"] & sets["heldout"])
    overlap_sh = len(sets["search"] & sets["heldout"])
    if overlap_cs or overlap_ch or overlap_sh:
        raise RuntimeError(
            f"Split overlap detected: calib∩search={overlap_cs} "
            f"calib∩heldout={overlap_ch} search∩heldout={overlap_sh}"
        )
    log.append("6. Verified splits are disjoint (no shared documents)")
    log.append("7. Hard rule: heldout.txt is for validation ONLY — never for search")

    notes = [
        "Token counts are estimates (chars/4), not tokenizer-exact.",
        "Optimizer (Steps 12–13) must not read heldout.txt.",
        f"corpus_id={CORPUS_ID}; increase --target-tokens for production (300k–1.5M).",
    ]
    if specialty_domain:
        notes.append(f"Included specialty domain prompts: {specialty_domain}")

    result = CorpusResult(
        model_ref=model_ref,
        corpus_id=CORPUS_ID,
        chat_template=chat_template,
        specialty_domain=specialty_domain,
        seed=seed,
        target_tokens=target_tokens,
        splits=splits,
        n_documents=len(docs),
        n_calib=len(parts["calib"]),
        n_search=len(parts["search"]),
        n_heldout=len(parts["heldout"]),
        tokens_est_total=sum(meta_files[k]["tokens_est"] for k in meta_files),
        tokens_est_calib=meta_files["calib"]["tokens_est"],
        tokens_est_search=meta_files["search"]["tokens_est"],
        tokens_est_heldout=meta_files["heldout"]["tokens_est"],
        chars_total=sum(meta_files[k]["chars"] for k in meta_files),
        domain_counts=domain_counts,
        files={k: meta_files[k]["path"] for k in meta_files},
        steps_log=log,
        notes=notes,
    )

    manifest = {
        **result.summary_dict(),
        "file_meta": meta_files,
    }
    return result, manifest
