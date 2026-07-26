"""Step 04 — classify tensors by role / depth / quantizable."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import re
from collections import Counter


# --- from classify/types.py ---
@dataclass
class ClassifiedTensor:
    index: int
    name: str
    shape: list[int]
    dtype: str
    n_elements: int
    nbytes: int
    role: str
    layer: int | None
    depth: str | None
    group_id: str
    quantizable: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClassificationResult:
    n_tensors: int
    n_layers: int
    role_summary: dict[str, int]
    depth_summary: dict[str, int]
    quantizable_summary: dict[str, int]
    group_summary: dict[str, int]
    other_names: list[str]
    coverage: float
    tensors: list[ClassifiedTensor]
    steps_log: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_tensors": self.n_tensors,
            "n_layers": self.n_layers,
            "role_summary": self.role_summary,
            "depth_summary": self.depth_summary,
            "quantizable_summary": self.quantizable_summary,
            "group_summary": self.group_summary,
            "other_names": self.other_names,
            "coverage": self.coverage,
            "tensors": [t.to_dict() for t in self.tensors],
            "steps_log": self.steps_log,
            "notes": self.notes,
        }

    def summary_dict(self) -> dict[str, Any]:
        return {
            "n_tensors": self.n_tensors,
            "n_layers": self.n_layers,
            "role_summary": self.role_summary,
            "depth_summary": self.depth_summary,
            "quantizable_summary": self.quantizable_summary,
            "group_summary": self.group_summary,
            "other_names": self.other_names,
            "coverage": self.coverage,
            "sample_tensors": [t.to_dict() for t in self.tensors[:20]],
            "steps_log": self.steps_log,
            "notes": self.notes,
        }

# --- from classify/classify.py ---
ROLE_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"token_embd|embed_tokens|tok_embeddings|token_embd", re.I), "embedding"),
    # output.weight (lm head) but NOT output_norm
    (re.compile(r"(^|\.)output\.weight$|(^|\.)lm_head(\.|$)", re.I), "lm_head"),
    # All norms (GGUF + HF)
    (
        re.compile(
            r"(_norm|layernorm|rms_norm|input_layernorm|post_attention_layernorm|"
            r"post_ffw_norm|post_attention_norm|attn_q_norm|attn_k_norm|"
            r"output_norm|ffn_norm|attn_norm)(\.|$)",
            re.I,
        ),
        "norm",
    ),
    (re.compile(r"attn_q|q_proj|(^|\.)wq(\.|$)", re.I), "attn_q"),
    (re.compile(r"attn_k|k_proj|(^|\.)wk(\.|$)", re.I), "attn_k"),
    (re.compile(r"attn_v|v_proj|(^|\.)wv(\.|$)", re.I), "attn_v"),
    (re.compile(r"attn_output|o_proj|(^|\.)wo(\.|$)|attn_o", re.I), "attn_o"),
    (re.compile(r"ffn_gate|gate_proj|(^|\.)w1(\.|$)", re.I), "ffn_gate"),
    (re.compile(r"ffn_up|up_proj|(^|\.)w3(\.|$)", re.I), "ffn_up"),
    (re.compile(r"ffn_down|down_proj|(^|\.)w2(\.|$)", re.I), "ffn_down"),
    (re.compile(r"experts?\.", re.I), "ffn_exps"),
    (re.compile(r"(^|\.)(router|ffn_gate_inp)(\.|$)", re.I), "router"),
    (re.compile(r"ssm_|in_proj|out_proj|conv1d|x_proj|dt_proj", re.I), "ssm"),
]

_LAYER_RE = re.compile(r"(?:blk|layers?)[.\[](\d+)", re.I)

NON_QUANTIZABLE = {"norm"}


def depth_bucket(layer: int | None, n_layers: int) -> str | None:
    if layer is None or n_layers <= 0:
        return None
    if n_layers < 3:
        return "all"
    a = n_layers // 3
    b = 2 * n_layers // 3
    if layer < a:
        return "early"
    if layer < b:
        return "middle"
    return "late"


def classify_name(name: str) -> str:
    for pat, role in ROLE_RULES:
        if pat.search(name):
            return role
    return "other"


def classify_tensors(
    tensors: list[dict[str, Any]],
    *,
    n_layers: int | None = None,
) -> ClassificationResult:
    log: list[str] = []
    log.append(f"1. Classifying {len(tensors)} tensors from Step 03")

    if n_layers is None:
        layers = [t.get("layer") for t in tensors if t.get("layer") is not None]
        n_layers = (max(layers) + 1) if layers else 0
    log.append(f"2. Using n_layers={n_layers} for depth buckets (early/middle/late thirds)")

    classified: list[ClassifiedTensor] = []
    role_c: Counter[str] = Counter()
    depth_c: Counter[str] = Counter()
    quant_c: Counter[str] = Counter()
    group_c: Counter[str] = Counter()
    other_names: list[str] = []

    for t in tensors:
        name = t["name"]
        layer = t.get("layer")
        if layer is None:
            m = _LAYER_RE.search(name)
            layer = int(m.group(1)) if m else None

        role = classify_name(name)
        depth = depth_bucket(layer, n_layers) if layer is not None else "global"
        quantizable = role not in NON_QUANTIZABLE
        # embeddings / lm_head stay quantizable=True but often pinned later
        group_id = f"{role}@{depth}" if depth else role

        row = ClassifiedTensor(
            index=int(t.get("index", 0)),
            name=name,
            shape=list(t.get("shape") or []),
            dtype=str(t.get("dtype") or "unknown"),
            n_elements=int(t.get("n_elements") or 0),
            nbytes=int(t.get("nbytes") or 0),
            role=role,
            layer=layer,
            depth=depth,
            group_id=group_id,
            quantizable=quantizable,
        )
        classified.append(row)
        role_c[role] += 1
        depth_c[str(depth)] += 1
        quant_c["yes" if quantizable else "no"] += 1
        group_c[group_id] += 1
        if role == "other":
            other_names.append(name)

    classified.sort(key=lambda r: r.name)
    coverage = 1.0 - (len(other_names) / max(len(classified), 1))
    log.append(f"3. Role breakdown: {dict(role_c.most_common())}")
    log.append(f"4. Quantizable: {dict(quant_c)}  coverage(non-other)={coverage:.1%}")
    log.append(f"5. Probe groups (role@depth): {len(group_c)}")
    if other_names:
        log.append(f"6. Unmatched 'other' ({len(other_names)}): {other_names[:8]}")
    else:
        log.append("6. All tensors matched a known role")
    log.append("7. Classification complete — catalog assembly is Step 05")

    notes = [
        "norm tensors are marked quantizable=false (usually keep F16/F32).",
        "group_id = role@depth is the unit Step 12 will probe.",
    ]
    if coverage < 0.95:
        notes.append(
            f"Coverage {coverage:.1%} < 95% — review other_names and extend ROLE_RULES."
        )

    return ClassificationResult(
        n_tensors=len(classified),
        n_layers=n_layers,
        role_summary=dict(role_c.most_common()),
        depth_summary=dict(depth_c),
        quantizable_summary=dict(quant_c),
        group_summary=dict(sorted(group_c.items(), key=lambda kv: (-kv[1], kv[0]))),
        other_names=other_names,
        coverage=round(coverage, 4),
        tensors=classified,
        steps_log=log,
        notes=notes,
    )
