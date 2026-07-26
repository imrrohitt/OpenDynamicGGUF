"""User-facing quantization *targets* for dynamic per-tensor recipes.

You pick a size/quality class (e.g. Q4_K_M). OpenDynamicGGUF still builds a
**dynamic** mix: hard tensors stay higher precision, easy tensors go lower —
so accuracy is better than a flat single-type quant at the same file size.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class QuantFormat:
    id: str
    label: str
    short: str
    description: str
    size_hint: str  # approx bpw
    shrink_vs_bf16: str  # e.g. "~72% smaller"
    quality_hint: str
    technique: str  # short technique tag shown in the picker
    # llama-quantize fallback / family tag (lowercase)
    base_type: str
    # optimize: fraction of all-Q6_K estimated size
    budget_ratio: float
    # sensitivity baseline for Δbytes / ΔKLD
    baseline_type: str
    # types to probe in sensitivity
    probe_types: tuple[str, ...]
    recommended: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Ordered for the interactive picker.
# shrink_vs_bf16 ≈ 1 − (bpw / 16) relative to BF16/F16 weights.
FORMATS: tuple[QuantFormat, ...] = (
    QuantFormat(
        id="q4_k_m",
        label="Q4_K_M",
        short="Balanced",
        description="Dynamic K-quant mix aimed at Q4_K_M size — best everyday default.",
        size_hint="~4.0–4.5 bpw",
        shrink_vs_bf16="~72% smaller",
        quality_hint="Strong everyday; dynamic mix beats flat Q4",
        technique="Dynamic K-quant",
        base_type="q4_k_m",
        budget_ratio=0.72,
        baseline_type="Q6_K",
        probe_types=("Q3_K", "Q4_K", "Q5_K", "Q6_K"),
        recommended=True,
    ),
    QuantFormat(
        id="q5_k_m",
        label="Q5_K_M",
        short="Higher quality",
        description="Larger budget — more tensors stay Q5_K/Q6_K where it matters.",
        size_hint="~5.0–5.5 bpw",
        shrink_vs_bf16="~66% smaller",
        quality_hint="Closer to BF16 than Q4 class",
        technique="Dynamic K-quant",
        base_type="q5_k_m",
        budget_ratio=0.85,
        baseline_type="Q6_K",
        probe_types=("Q4_K", "Q5_K", "Q6_K", "Q8_0"),
    ),
    QuantFormat(
        id="q3_k_m",
        label="Q3_K_M",
        short="Compact",
        description="Aggressive size target — easy groups drop to Q3_K; hard groups protected.",
        size_hint="~3.0–3.5 bpw",
        shrink_vs_bf16="~78% smaller",
        quality_hint="Good for edge / low RAM",
        technique="Dynamic K-quant",
        base_type="q3_k_m",
        budget_ratio=0.55,
        baseline_type="Q6_K",
        probe_types=("Q2_K", "Q3_K", "Q4_K", "Q5_K"),
    ),
    QuantFormat(
        id="q4_k_s",
        label="Q4_K_S",
        short="Smaller Q4",
        description="Q4_K_S-class footprint — slightly smaller/faster than Q4_K_M target.",
        size_hint="~3.8–4.2 bpw",
        shrink_vs_bf16="~75% smaller",
        quality_hint="A bit tighter than Q4_K_M",
        technique="Dynamic K-quant",
        base_type="q4_k_s",
        budget_ratio=0.66,
        baseline_type="Q6_K",
        probe_types=("Q3_K", "Q4_K", "Q5_K", "Q6_K"),
    ),
    QuantFormat(
        id="q5_k_s",
        label="Q5_K_S",
        short="Leaner Q5",
        description="Q5_K_S-class target — quality-leaning but smaller than Q5_K_M.",
        size_hint="~4.8–5.2 bpw",
        shrink_vs_bf16="~68% smaller",
        quality_hint="Quality-leaning, leaner file",
        technique="Dynamic K-quant",
        base_type="q5_k_s",
        budget_ratio=0.80,
        baseline_type="Q6_K",
        probe_types=("Q4_K", "Q5_K", "Q6_K", "Q8_0"),
    ),
    QuantFormat(
        id="q6_k",
        label="Q6_K",
        short="High fidelity",
        description="Near-reference dynamic mix; little compression vs Q6_K.",
        size_hint="~6.0–6.5 bpw",
        shrink_vs_bf16="~60% smaller",
        quality_hint="Minimal quality loss",
        technique="Dynamic K-quant",
        base_type="q6_k",
        budget_ratio=0.95,
        baseline_type="Q8_0",
        probe_types=("Q5_K", "Q6_K", "Q8_0"),
    ),
    QuantFormat(
        id="q8_0",
        label="Q8_0",
        short="Near lossless",
        description="Largest practical GGUF — mostly Q8_0 with light dynamic pins.",
        size_hint="~8.0–8.5 bpw",
        shrink_vs_bf16="~50% smaller",
        quality_hint="Closest to source weights",
        technique="Dynamic high-bit",
        base_type="q8_0",
        budget_ratio=1.15,
        baseline_type="Q8_0",
        probe_types=("Q6_K", "Q8_0"),
    ),
    QuantFormat(
        id="q2_k",
        label="Q2_K",
        short="Smallest",
        description="Maximum compression — expect larger ΔKLD; for tiny footprints.",
        size_hint="~2.0–2.5 bpw",
        shrink_vs_bf16="~85% smaller",
        quality_hint="Size-critical only",
        technique="Dynamic K-quant",
        base_type="q2_k",
        budget_ratio=0.42,
        baseline_type="Q6_K",
        probe_types=("Q2_K", "Q3_K", "Q4_K"),
    ),
    QuantFormat(
        id="iq4_xs",
        label="IQ4_XS",
        short="I-quant 4-bit",
        description="Importance-matrix (I-quant) 4-bit class — best with a real imatrix.",
        size_hint="~4.25 bpw",
        shrink_vs_bf16="~73% smaller",
        quality_hint="Better accuracy/size than flat Q4",
        technique="Dynamic + imatrix (I-quant)",
        base_type="iq4_xs",
        budget_ratio=0.70,
        baseline_type="Q6_K",
        probe_types=("Q3_K", "Q4_K", "Q5_K", "Q6_K"),
    ),
    QuantFormat(
        id="iq4_nl",
        label="IQ4_NL",
        short="I-quant non-linear",
        description="IQ4_NL-class — non-linear 4-bit with imatrix; strong mid-size quality.",
        size_hint="~4.5 bpw",
        shrink_vs_bf16="~72% smaller",
        quality_hint="Strong mid-size with imatrix",
        technique="Dynamic + imatrix (I-quant)",
        base_type="iq4_nl",
        budget_ratio=0.74,
        baseline_type="Q6_K",
        probe_types=("Q3_K", "Q4_K", "Q5_K", "Q6_K"),
    ),
    QuantFormat(
        id="iq3_m",
        label="IQ3_M",
        short="I-quant 3-bit",
        description="IQ3_M-class — aggressive imatrix 3-bit; protect hard tensors dynamically.",
        size_hint="~3.3 bpw",
        shrink_vs_bf16="~79% smaller",
        quality_hint="Smaller than Q3 with better recovery via imatrix",
        technique="Dynamic + imatrix (I-quant)",
        base_type="iq3_m",
        budget_ratio=0.52,
        baseline_type="Q6_K",
        probe_types=("Q2_K", "Q3_K", "Q4_K", "Q5_K"),
    ),
    QuantFormat(
        id="iq2_xxs",
        label="IQ2_XXS",
        short="I-quant tiny",
        description="IQ2_XXS-class — extreme compression; dynamic pins keep critical tensors alive.",
        size_hint="~2.1 bpw",
        shrink_vs_bf16="~87% smaller",
        quality_hint="Extreme size; expect quality tradeoffs",
        technique="Dynamic + imatrix (I-quant)",
        base_type="iq2_xxs",
        budget_ratio=0.38,
        baseline_type="Q6_K",
        probe_types=("Q2_K", "Q3_K", "Q4_K"),
    ),
    QuantFormat(
        id="q4_0",
        label="Q4_0",
        short="Legacy 4-bit",
        description="Classic Q4_0 family target — widely compatible; dynamic mix still applies.",
        size_hint="~4.5 bpw",
        shrink_vs_bf16="~72% smaller",
        quality_hint="Compatible baseline; K/I-quants usually better",
        technique="Dynamic legacy quant",
        base_type="q4_0",
        budget_ratio=0.70,
        baseline_type="Q6_K",
        probe_types=("Q3_K", "Q4_K", "Q5_K", "Q6_K"),
    ),
)

FORMATS_BY_ID: dict[str, QuantFormat] = {f.id: f for f in FORMATS}
DEFAULT_FORMAT_ID = "q4_k_m"

# Common aliases users type
_ALIASES: dict[str, str] = {
    "q4": "q4_k_m",
    "q4_k": "q4_k_m",
    "q4km": "q4_k_m",
    "q4ks": "q4_k_s",
    "q5": "q5_k_m",
    "q5_k": "q5_k_m",
    "q5km": "q5_k_m",
    "q5ks": "q5_k_s",
    "q3": "q3_k_m",
    "q3_k": "q3_k_m",
    "q3km": "q3_k_m",
    "q6": "q6_k",
    "q8": "q8_0",
    "iq4": "iq4_xs",
    "iq3": "iq3_m",
    "iq2": "iq2_xxs",
    "balanced": "q4_k_m",
    "default": "q4_k_m",
    "small": "q3_k_m",
    "tiny": "q2_k",
    "quality": "q5_k_m",
    "hq": "q6_k",
    "best": "q8_0",
}


def normalize_format_id(value: str) -> str:
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    key = _ALIASES.get(key, key)
    if key not in FORMATS_BY_ID:
        known = ", ".join(f.id for f in FORMATS)
        raise ValueError(
            f"Unknown quant format {value!r}. Choose one of: {known}"
        )
    return key


def get_format(value: str | None = None) -> QuantFormat:
    if value is None:
        return FORMATS_BY_ID[DEFAULT_FORMAT_ID]
    return FORMATS_BY_ID[normalize_format_id(value)]


def format_choices() -> list[str]:
    return [f.id for f in FORMATS]


def list_formats_rows() -> list[dict[str, str]]:
    rows = []
    for f in FORMATS:
        rows.append(
            {
                "id": f.id,
                "label": f.label,
                "short": f"{f.short} ★" if f.recommended else f.short,
                "size": f.size_hint,
                "shrink": f.shrink_vs_bf16,
                "quality": f.quality_hint,
                "technique": f.technique,
                "description": f.description,
            }
        )
    return rows
