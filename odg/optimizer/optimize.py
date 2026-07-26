"""
Step 13 — Greedy recipe optimizer under a size budget.

maximize bytes_saved / ΔKLD (from sensitivity table) subject to size ≤ budget.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from odg.sensitivity.proxy import BYTES_PER_ELEM, estimate_group_nbytes

from .types import OptimizeResult

# Precision ladder high → low (downgrade direction)
LADDER = ["Q8_0", "Q6_K", "Q5_K", "Q4_K", "Q3_K", "Q2_K"]

# Role floor pins (still overridable via --no-pins)
DEFAULT_PINS: dict[str, str] = {
    "embedding": "Q8_0",
    "lm_head": "Q8_0",
    "attn_v": "Q5_K",
}


def _ladder_index(q: str) -> int:
    u = q.upper()
    if u not in LADDER:
        raise KeyError(f"Unknown quant on ladder: {q}")
    return LADDER.index(u)


def _min_quant(a: str, b: str) -> str:
    """Higher precision wins (lower ladder index)."""
    return a if _ladder_index(a) <= _ladder_index(b) else b


def _group_n_elements(group: dict[str, Any], tensors: dict[str, Any]) -> int:
    total = 0
    for name in group.get("tensor_names") or []:
        t = tensors.get(name) or {}
        total += int(t.get("n_elements") or 0)
    return total


def _build_row_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    idx: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        idx[(r["group_id"], str(r["probe"]).upper())] = r
    return idx


def _estimate_total_bytes(
    assignments: dict[str, str],
    groups: dict[str, Any],
    tensors: dict[str, Any],
) -> int:
    total = 0
    assigned = set()
    for gid, q in assignments.items():
        g = groups.get(gid) or {}
        n = _group_n_elements(g, tensors)
        total += estimate_group_nbytes(n, q)
        assigned.add(gid)
    # Non-quantizable / unassigned: keep catalog nbytes (usually F32 norms)
    for name, t in tensors.items():
        gid = t.get("group_id")
        if gid in assigned and t.get("quantizable", True):
            continue
        total += int(t.get("nbytes") or 0)
    return total


def _predicted_kld(
    assignments: dict[str, str],
    row_index: dict[tuple[str, str], dict[str, Any]],
    baseline: str,
) -> float:
    """Sum per-group ΔKLD vs baseline (proxy-additive)."""
    s = 0.0
    for gid, q in assignments.items():
        if q.upper() == baseline.upper():
            continue
        row = row_index.get((gid, q.upper()))
        if row:
            s += float(row.get("delta_kld") or 0.0)
        else:
            # interpolate crudely from nearest
            s += 0.01
    return s


def _downgrade_gain(
    gid: str,
    cur_q: str,
    next_q: str,
    row_index: dict[tuple[str, str], dict[str, Any]],
    groups: dict[str, Any],
    tensors: dict[str, Any],
) -> tuple[float, float, float]:
    """
    Returns (efficiency, delta_bytes, delta_kld_inc) for cur→next downgrade.
    Prefer sensitivity rows; fall back to size/KLD estimates.
    """
    g = groups.get(gid) or {}
    n = _group_n_elements(g, tensors)
    bytes_cur = estimate_group_nbytes(n, cur_q)
    bytes_next = estimate_group_nbytes(n, next_q)
    delta_bytes = bytes_cur - bytes_next

    row_cur = row_index.get((gid, cur_q.upper()))
    row_next = row_index.get((gid, next_q.upper()))
    kld_cur = float(row_cur["delta_kld"]) if row_cur else 0.0
    kld_next = float(row_next["delta_kld"]) if row_next else kld_cur + 0.01
    d_kld = max(kld_next - kld_cur, 1e-9)
    eff = delta_bytes / d_kld if delta_bytes > 0 else 0.0
    return eff, float(delta_bytes), float(d_kld)


def greedy_optimize(
    *,
    catalog: dict[str, Any],
    sensitivity_rows: list[dict[str, Any]],
    budget_bytes: int,
    start_type: str = "Q6_K",
    pins: dict[str, str] | None = None,
    use_pins: bool = True,
) -> dict[str, Any]:
    """
    Start high, greedily downgrade best efficiency until size ≤ budget.
    """
    pins = dict(DEFAULT_PINS) if use_pins else {}
    groups = catalog.get("groups") or {}
    tensors = catalog.get("tensors") or {}
    row_index = _build_row_index(sensitivity_rows)

    assignments: dict[str, str] = {}
    floors: dict[str, str] = {}

    for gid, g in groups.items():
        if not g.get("quantizable", True):
            continue
        role = str(g.get("role") or "")
        q = start_type.upper()
        floor = pins.get(role)
        if floor:
            floors[gid] = floor.upper()
            # Never start below the role floor
            if _ladder_index(q) > _ladder_index(floor):
                q = floor.upper()
        assignments[gid] = q

    # pin_high hints from sensitivity → floor at Q5_K
    for r in sensitivity_rows:
        if r.get("decision_hint") == "pin_high" and r.get("probe") == "Q4_K":
            gid = r["group_id"]
            floors[gid] = _min_quant(floors.get(gid, "Q3_K"), "Q5_K")
            if _ladder_index(assignments.get(gid, start_type)) > _ladder_index(
                floors[gid]
            ):
                assignments[gid] = floors[gid]

    def size_now() -> int:
        return _estimate_total_bytes(assignments, groups, tensors)

    history: list[dict[str, Any]] = []
    # Greedy loop
    safety = 0
    while size_now() > budget_bytes and safety < 10_000:
        safety += 1
        best = None  # (eff, gid, next_q, d_bytes, d_kld)
        for gid, cur in assignments.items():
            floor = floors.get(gid)
            cur_i = _ladder_index(cur)
            if cur_i >= len(LADDER) - 1:
                continue
            next_q = LADDER[cur_i + 1]
            if floor and _ladder_index(next_q) > _ladder_index(floor):
                continue
            # Only consider probes that exist in sensitivity when possible
            if (gid, next_q) not in row_index and next_q not in BYTES_PER_ELEM:
                continue
            eff, d_b, d_k = _downgrade_gain(
                gid, cur, next_q, row_index, groups, tensors
            )
            if d_b <= 0:
                continue
            cand = (eff, gid, next_q, d_b, d_k)
            if best is None or cand[0] > best[0]:
                best = cand
        if best is None:
            break
        _, gid, next_q, d_b, d_k = best
        assignments[gid] = next_q
        history.append(
            {
                "group_id": gid,
                "to": next_q,
                "delta_bytes": d_b,
                "delta_kld_inc": d_k,
                "efficiency": best[0],
                "size_after": size_now(),
            }
        )

    est = size_now()
    pred_kld = _predicted_kld(assignments, row_index, start_type)
    return {
        "assignments": assignments,
        "floors": floors,
        "estimated_bytes": est,
        "predicted_delta_kld": pred_kld,
        "meets_budget": est <= budget_bytes,
        "history": history,
        "budget_bytes": budget_bytes,
        "start_type": start_type,
    }


def _yaml_escape(s: str) -> str:
    if any(c in s for c in ":#{}[]|&*!?>'%@`,"):
        return json.dumps(s)
    return s


def render_recipe_yaml(
    *,
    model_ref: str,
    hf_repo_id: str | None,
    gguf_sha256: str | None,
    imatrix_sha256: str | None,
    corpus_id: str | None,
    budget_bytes: int,
    base_type: str,
    assignments: dict[str, str],
    groups: dict[str, Any],
    estimated_bytes: int,
    predicted_delta_kld: float,
    method: str,
) -> str:
    overrides_lines = []
    for gid, q in sorted(assignments.items()):
        g = groups.get(gid) or {}
        regex = g.get("tensor_type_regex")
        # Prefer regex from first tensor names pattern stored in sensitivity — rebuild
        names = g.get("tensor_names") or []
        if names:
            from odg.sensitivity.proxy import tensor_type_regex

            regex = tensor_type_regex(g)
        else:
            regex = gid
        overrides_lines.append(f'  "{regex}": {q.lower()}')

    lines = [
        "# OpenDynamicGGUF recipe — odg/recipe/v1",
        "schema: odg/recipe/v1",
        "model:",
        f"  source: {_yaml_escape(model_ref)}",
        f"  hf_repo_id: {_yaml_escape(hf_repo_id) if hf_repo_id else 'null'}",
        f"  gguf_sha256: \"{gguf_sha256 or ''}\"",
        "calibration:",
        f"  corpus_id: {_yaml_escape(corpus_id or 'odg-corpus-v1')}",
        f"  imatrix_sha256: \"{imatrix_sha256 or ''}\"",
        "  splits: { calib: 0.6, search: 0.2, heldout: 0.2, seed: 42 }",
        "budget:",
        f"  target_size_bytes: {budget_bytes}",
        f"  target_size_mb: {budget_bytes / (1024 * 1024):.2f}",
        f"base_type: {base_type.lower()}",
        "overrides:",
        *overrides_lines,
        "estimate:",
        f"  size_bytes: {estimated_bytes}",
        f"  size_mb: {estimated_bytes / (1024 * 1024):.2f}",
        f"  predicted_mean_delta_kld: {predicted_delta_kld:.6f}",
        f"  method: {method}",
        "validation: {}",
        "",
    ]
    return "\n".join(lines)


def render_tensor_type_file(
    assignments: dict[str, str],
    groups: dict[str, Any],
) -> str:
    """llama-quantize --tensor-type-file format: REGEX=TYPE per line."""
    from odg.sensitivity.proxy import tensor_type_regex

    lines = ["# tensor-type-file generated by odg optimize"]
    for gid, q in sorted(assignments.items()):
        g = groups.get(gid) or {}
        regex = tensor_type_regex(g) if g.get("tensor_names") else gid
        lines.append(f"{regex}={q.lower()}")
    lines.append("")
    return "\n".join(lines)


def default_budget_bytes(catalog: dict[str, Any], *, ratio: float = 0.72) -> int:
    """
    Budget as a fraction of all-Q6_K size, but never below the pinned floor
    (embd Q8 + attn_v Q5 + rest Q3) so greedy can actually meet it.
    """
    groups = catalog.get("groups") or {}
    tensors = catalog.get("tensors") or {}
    q6 = {
        gid: "Q6_K"
        for gid, g in groups.items()
        if g.get("quantizable", True)
    }
    full = _estimate_total_bytes(q6, groups, tensors)
    # Minimum achievable with default pins + Q3 elsewhere
    floor_assign = {}
    for gid, g in groups.items():
        if not g.get("quantizable", True):
            continue
        role = str(g.get("role") or "")
        floor_assign[gid] = DEFAULT_PINS.get(role, "Q3_K")
    floor_bytes = _estimate_total_bytes(floor_assign, groups, tensors)
    target = int(full * ratio)
    # Leave a little headroom above the pin floor
    return max(1, max(target, int(floor_bytes * 1.02)))


def optimize_recipes(
    *,
    model_ref: str,
    out_dir: Path,
    catalog: dict[str, Any],
    sensitivity: dict[str, Any],
    budget_bytes: int | None = None,
    budget_ratio: float = 0.72,
    hf_repo_id: str | None = None,
    gguf_sha256: str | None = None,
    imatrix_sha256: str | None = None,
    corpus_id: str | None = None,
    use_pins: bool = True,
) -> OptimizeResult:
    log: list[str] = []
    notes: list[str] = []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pareto_dir = out_dir / "pareto"
    pareto_dir.mkdir(exist_ok=True)

    rows = sensitivity.get("rows") or []
    if not rows:
        raise ValueError("sensitivity table has no rows")

    if budget_bytes is None:
        budget_bytes = default_budget_bytes(catalog, ratio=budget_ratio)
        log.append(
            f"1. Budget from ratio={budget_ratio:.2f} → {budget_bytes} bytes "
            f"({budget_bytes / (1024**2):.1f} MiB)"
        )
    else:
        log.append(
            f"1. Budget fixed → {budget_bytes} bytes "
            f"({budget_bytes / (1024**2):.1f} MiB)"
        )

    log.append(f"2. Sensitivity rows={len(rows)} method={sensitivity.get('method')}")
    log.append("3. Greedy downgrade from Q6_K with role pins")

    primary = greedy_optimize(
        catalog=catalog,
        sensitivity_rows=rows,
        budget_bytes=budget_bytes,
        start_type="Q6_K",
        use_pins=use_pins,
    )
    log.append(
        f"4. Primary recipe size={primary['estimated_bytes']} "
        f"meets_budget={primary['meets_budget']} "
        f"pred_ΔKLD={primary['predicted_delta_kld']:.4f} "
        f"steps={len(primary['history'])}"
    )

    groups = catalog.get("groups") or {}
    method = "greedy_knapsack_v1"

    recipe_yaml = render_recipe_yaml(
        model_ref=model_ref,
        hf_repo_id=hf_repo_id,
        gguf_sha256=gguf_sha256,
        imatrix_sha256=imatrix_sha256,
        corpus_id=corpus_id,
        budget_bytes=budget_bytes,
        base_type="Q6_K",
        assignments=primary["assignments"],
        groups=groups,
        estimated_bytes=primary["estimated_bytes"],
        predicted_delta_kld=primary["predicted_delta_kld"],
        method=method,
    )
    recipe_path = out_dir / "recipe.yaml"
    recipe_path.write_text(recipe_yaml, encoding="utf-8")

    tt = render_tensor_type_file(primary["assignments"], groups)
    tt_path = out_dir / "recipe.tt"
    tt_path.write_text(tt, encoding="utf-8")

    # Pareto: optimize under several budgets
    q6_size = default_budget_bytes(catalog, ratio=1.0)
    pareto_targets = sorted(
        {
            int(q6_size * r)
            for r in (0.55, 0.65, 0.72, 0.80, 0.90, 1.0)
        }
        | {budget_bytes}
    )
    pareto_paths: list[str] = []
    pareto_summary = []
    for i, b in enumerate(pareto_targets):
        alt = greedy_optimize(
            catalog=catalog,
            sensitivity_rows=rows,
            budget_bytes=b,
            start_type="Q6_K",
            use_pins=use_pins,
        )
        name = f"pareto-{i:02d}-{b // 1024}k.yaml"
        y = render_recipe_yaml(
            model_ref=model_ref,
            hf_repo_id=hf_repo_id,
            gguf_sha256=gguf_sha256,
            imatrix_sha256=imatrix_sha256,
            corpus_id=corpus_id,
            budget_bytes=b,
            base_type="Q6_K",
            assignments=alt["assignments"],
            groups=groups,
            estimated_bytes=alt["estimated_bytes"],
            predicted_delta_kld=alt["predicted_delta_kld"],
            method=method,
        )
        p = pareto_dir / name
        p.write_text(y, encoding="utf-8")
        pareto_paths.append(str(p))
        pareto_summary.append(
            {
                "path": str(p),
                "budget_bytes": b,
                "estimated_bytes": alt["estimated_bytes"],
                "predicted_delta_kld": alt["predicted_delta_kld"],
                "meets_budget": alt["meets_budget"],
            }
        )

    log.append(f"5. Wrote recipe.yaml + recipe.tt + {len(pareto_paths)} Pareto recipes")
    notes.append(
        "Assignments from sensitivity table (proxy or measured). "
        "Export with Step 14 using recipe.tt."
    )
    if not primary["meets_budget"]:
        notes.append(
            "Could not fully meet budget (hit pin floors). "
            "Relax pins or raise --budget-mb."
        )

    (out_dir / "optimize_manifest.json").write_text(
        json.dumps(
            {
                "primary": {
                    "assignments": primary["assignments"],
                    "estimated_bytes": primary["estimated_bytes"],
                    "predicted_delta_kld": primary["predicted_delta_kld"],
                    "meets_budget": primary["meets_budget"],
                    "history": primary["history"],
                },
                "pareto": pareto_summary,
                "budget_bytes": budget_bytes,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return OptimizeResult(
        model_ref=model_ref,
        method=method,
        budget_bytes=budget_bytes,
        estimated_bytes=primary["estimated_bytes"],
        predicted_delta_kld=primary["predicted_delta_kld"],
        n_groups=len(primary["assignments"]),
        recipe_path=str(recipe_path),
        tensor_type_file=str(tt_path),
        pareto_paths=pareto_paths,
        assignments=primary["assignments"],
        steps_log=log,
        notes=notes,
    )
