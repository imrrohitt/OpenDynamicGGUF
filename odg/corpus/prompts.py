"""Built-in mixed-domain prompt banks for Step 07 (offline, no downloads)."""

from __future__ import annotations

from typing import Any

# Each entry: domain, user, optional assistant (for richer calib activations)


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
