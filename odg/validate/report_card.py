"""
Quantization Report Card — detailed per-layer / per-group compression report.

Writes:
  quantization_report_card.html
  quantization_report_card.md
  quantization_report_card.json
"""

from __future__ import annotations

import html as html_lib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from odg.sensitivity.proxy import estimate_group_nbytes

BASELINE = "Q6_K"


def _mb(n: int | float | None) -> float:
    if not n:
        return 0.0
    return float(n) / (1024 * 1024)


def _group_n_elements(group: dict[str, Any], tensors: dict[str, Any]) -> int:
    total = 0
    for name in group.get("tensor_names") or []:
        t = tensors.get(name) or {}
        total += int(t.get("n_elements") or 0)
    return total


def _parse_layers_from_group(group: dict[str, Any]) -> list[int]:
    layers: list[int] = []
    for name in group.get("tensor_names") or []:
        # blk.N....
        if name.startswith("blk."):
            parts = name.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                layers.append(int(parts[1]))
    return sorted(set(layers))


def build_report_card_data(
    *,
    model_ref: str,
    catalog: dict[str, Any],
    assignments: dict[str, str],
    sensitivity_rows: list[dict[str, Any]] | None = None,
    optimize_manifest: dict[str, Any] | None = None,
    validate_payload: dict[str, Any] | None = None,
    resolve_descriptor: dict[str, Any] | None = None,
    baseline: str = BASELINE,
) -> dict[str, Any]:
    tensors = catalog.get("tensors") or {}
    groups = catalog.get("groups") or {}
    n_layers = int(catalog.get("n_layers") or 0)
    if not n_layers:
        # infer
        mx = -1
        for t in tensors.values():
            if t.get("layer") is not None:
                mx = max(mx, int(t["layer"]))
        n_layers = mx + 1 if mx >= 0 else 0

    row_index: dict[tuple[str, str], dict[str, Any]] = {}
    for r in sensitivity_rows or []:
        row_index[(r["group_id"], str(r["probe"]).upper())] = r

    primary = (optimize_manifest or {}).get("primary") or {}
    history = primary.get("history") or []

    # --- per-group rows ---
    group_rows: list[dict[str, Any]] = []
    total_base = 0
    total_final = 0
    total_kld = 0.0

    for gid, g in sorted(groups.items()):
        if not g.get("quantizable", True):
            continue
        q = (assignments.get(gid) or baseline).upper()
        n_elem = _group_n_elements(g, tensors)
        base_b = estimate_group_nbytes(n_elem, baseline)
        final_b = estimate_group_nbytes(n_elem, q)
        saved = base_b - final_b
        ratio = (final_b / base_b) if base_b else 1.0
        compress_pct = (1.0 - ratio) * 100.0
        sens = row_index.get((gid, q)) or {}
        delta_kld = float(sens.get("delta_kld") or 0.0)
        layers = _parse_layers_from_group(g)
        total_base += base_b
        total_final += final_b
        total_kld += delta_kld
        group_rows.append(
            {
                "group_id": gid,
                "role": g.get("role"),
                "depth": g.get("depth"),
                "layers": layers,
                "n_tensors": g.get("n_tensors"),
                "n_elements": n_elem,
                "baseline": baseline,
                "assigned": q,
                "bytes_baseline": base_b,
                "bytes_final": final_b,
                "bytes_saved": saved,
                "compress_pct": round(compress_pct, 2),
                "delta_kld": delta_kld,
                "decision_hint": sens.get("decision_hint"),
                "tensor_names": list(g.get("tensor_names") or []),
            }
        )

    # Non-quantizable overhead (norms etc.)
    fixed_bytes = 0
    fixed_tensors = []
    for name, t in tensors.items():
        if t.get("quantizable", True):
            continue
        nb = int(t.get("nbytes") or 0)
        fixed_bytes += nb
        fixed_tensors.append(
            {
                "name": name,
                "role": t.get("role"),
                "layer": t.get("layer"),
                "dtype": t.get("dtype"),
                "nbytes": nb,
            }
        )

    # --- per-layer matrix ---
    roles_order = [
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_o",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
    ]
    layer_cards: list[dict[str, Any]] = []
    for layer in range(n_layers):
        # depth bucket
        if n_layers <= 1:
            depth = "global"
        else:
            third = n_layers / 3.0
            if layer < third:
                depth = "early"
            elif layer < 2 * third:
                depth = "middle"
            else:
                depth = "late"

        roles: dict[str, Any] = {}
        layer_base = 0
        layer_final = 0
        for role in roles_order:
            gid = f"{role}@{depth}"
            # Find group that contains this layer+role
            match = None
            for gr in group_rows:
                if gr["role"] == role and layer in (gr["layers"] or []):
                    match = gr
                    break
            if match is None:
                # try exact depth group
                for gr in group_rows:
                    if gr["group_id"] == gid:
                        match = gr
                        break
            if match is None:
                roles[role] = None
                continue
            # Per-layer share: divide group bytes by n_tensors in group
            n_t = max(int(match["n_tensors"] or 1), 1)
            b_base = match["bytes_baseline"] // n_t
            b_final = match["bytes_final"] // n_t
            layer_base += b_base
            layer_final += b_final
            roles[role] = {
                "group_id": match["group_id"],
                "assigned": match["assigned"],
                "bytes_baseline": b_base,
                "bytes_final": b_final,
                "bytes_saved": b_base - b_final,
                "compress_pct": round(
                    (1.0 - (b_final / b_base)) * 100.0 if b_base else 0.0, 2
                ),
                "delta_kld_group": match["delta_kld"],
            }

        # Norms on this layer (F32, not quantized)
        norms = [
            ft
            for ft in fixed_tensors
            if ft.get("layer") == layer and ft.get("role") == "norm"
        ]
        layer_cards.append(
            {
                "layer": layer,
                "depth": depth,
                "roles": roles,
                "bytes_baseline": layer_base,
                "bytes_final": layer_final,
                "bytes_saved": layer_base - layer_final,
                "compress_pct": round(
                    (1.0 - (layer_final / layer_base)) * 100.0 if layer_base else 0.0,
                    2,
                ),
                "norms_kept_f32": norms,
            }
        )

    # --- role summary ---
    by_role: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "groups": 0,
            "bytes_baseline": 0,
            "bytes_final": 0,
            "bytes_saved": 0,
            "assignments": defaultdict(int),
            "delta_kld_sum": 0.0,
        }
    )
    for gr in group_rows:
        role = str(gr["role"] or "other")
        br = by_role[role]
        br["groups"] += 1
        br["bytes_baseline"] += gr["bytes_baseline"]
        br["bytes_final"] += gr["bytes_final"]
        br["bytes_saved"] += gr["bytes_saved"]
        br["assignments"][gr["assigned"]] += 1
        br["delta_kld_sum"] += gr["delta_kld"]

    role_summary = []
    for role, br in sorted(by_role.items()):
        role_summary.append(
            {
                "role": role,
                "groups": br["groups"],
                "bytes_baseline": br["bytes_baseline"],
                "bytes_final": br["bytes_final"],
                "bytes_saved": br["bytes_saved"],
                "compress_pct": round(
                    (1.0 - br["bytes_final"] / br["bytes_baseline"]) * 100.0
                    if br["bytes_baseline"]
                    else 0.0,
                    2,
                ),
                "assignments": dict(br["assignments"]),
                "delta_kld_sum": br["delta_kld_sum"],
            }
        )

    # quant histogram
    quant_hist: dict[str, int] = defaultdict(int)
    for gr in group_rows:
        quant_hist[gr["assigned"]] += 1

    desc = resolve_descriptor or {}
    validate_payload = validate_payload or {}

    return {
        "title": "OpenDynamicGGUF — Quantization Report Card",
        "model_ref": model_ref,
        "architecture": {
            "family": desc.get("family") or catalog.get("source_backend"),
            "n_layers": n_layers,
            "n_tensors": catalog.get("n_tensors") or len(tensors),
            "n_quantizable": catalog.get("n_quantizable"),
            "n_groups": len(group_rows),
            "embedding_length": desc.get("embedding_length"),
            "parameter_count": desc.get("parameter_count"),
            "specialty_domain": desc.get("specialty_domain"),
            "chat_template": desc.get("chat_template"),
        },
        "baseline": baseline,
        "size": {
            "quantizable_baseline_bytes": total_base,
            "quantizable_final_bytes": total_final,
            "quantizable_saved_bytes": total_base - total_final,
            "quantizable_compress_pct": round(
                (1.0 - total_final / total_base) * 100.0 if total_base else 0.0, 2
            ),
            "fixed_nonquant_bytes": fixed_bytes,
            "estimated_total_bytes": total_final + fixed_bytes,
            "estimated_total_mb": round(_mb(total_final + fixed_bytes), 2),
            "baseline_total_mb": round(_mb(total_base + fixed_bytes), 2),
        },
        "quality": {
            "predicted_delta_kld_sum": total_kld,
            "optimize_predicted_delta_kld": primary.get("predicted_delta_kld"),
            "verdict": validate_payload.get("verdict"),
            "tier1": (validate_payload.get("tier1") or {}).get("metrics"),
        },
        "quant_histogram": dict(sorted(quant_hist.items())),
        "role_summary": role_summary,
        "groups": group_rows,
        "layers": layer_cards,
        "fixed_tensors": fixed_tensors,
        "optimizer_steps": len(history),
        "notes": [
            f"Compression % is vs {baseline} baseline (not vs BF16).",
            "ΔKLD values are from the sensitivity table (proxy or measured).",
            "Norms / non-quantizable tensors stay F32 and are listed under fixed.",
        ],
    }


def write_report_card_md(data: dict[str, Any], path: Path) -> None:
    arch = data["architecture"]
    size = data["size"]
    lines = [
        f"# {data['title']}",
        "",
        f"**Model:** `{data['model_ref']}`  ",
        f"**Verdict:** **{data['quality'].get('verdict') or 'n/a'}**  ",
        f"**Architecture:** {arch.get('family')} · "
        f"{arch.get('n_layers')} layers · "
        f"{arch.get('n_tensors')} tensors · "
        f"{arch.get('n_quantizable')} quantizable · "
        f"{arch.get('n_groups')} groups",
        "",
        "## Size summary",
        "",
        f"| | MiB |",
        f"|---|---:|",
        f"| Baseline ({data['baseline']}) quantizable | { _mb(size['quantizable_baseline_bytes']):.2f} |",
        f"| After recipe (quantizable) | { _mb(size['quantizable_final_bytes']):.2f} |",
        f"| Bytes saved | { _mb(size['quantizable_saved_bytes']):.2f} |",
        f"| Compression | **{size['quantizable_compress_pct']}%** |",
        f"| Fixed (norms etc.) | { _mb(size['fixed_nonquant_bytes']):.2f} |",
        f"| **Estimated total** | **{size['estimated_total_mb']:.2f}** |",
        "",
        f"Predicted Σ ΔKLD: `{data['quality'].get('predicted_delta_kld_sum', 0):.4f}`  ",
        f"Quant mix: `{data['quant_histogram']}`",
        "",
        "## Compression by role",
        "",
        "| Role | Groups | Baseline MiB | Final MiB | Saved MiB | Compress % | Assignments |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in data["role_summary"]:
        lines.append(
            f"| {r['role']} | {r['groups']} | {_mb(r['bytes_baseline']):.2f} | "
            f"{_mb(r['bytes_final']):.2f} | {_mb(r['bytes_saved']):.2f} | "
            f"{r['compress_pct']}% | `{r['assignments']}` |"
        )

    lines += [
        "",
        "## Per-layer report card",
        "",
        "Each cell: **quant type** (compress % vs baseline).",
        "",
        "| Layer | Depth | attn_q | attn_k | attn_v | attn_o | ffn_gate | ffn_up | ffn_down | Layer compress |",
        "|---:|---|---|---|---|---|---|---|---|---:|",
    ]
    for lc in data["layers"]:
        cells = []
        for role in [
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_o",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
        ]:
            info = (lc.get("roles") or {}).get(role)
            if not info:
                cells.append("—")
            else:
                cells.append(f"**{info['assigned']}** ({info['compress_pct']}%)")
        lines.append(
            f"| {lc['layer']} | {lc['depth']} | "
            + " | ".join(cells)
            + f" | **{lc['compress_pct']}%** |"
        )

    lines += [
        "",
        "## Per-group detail",
        "",
        "| Group | Layers | Assigned | Baseline MiB | Final MiB | Saved | Compress % | ΔKLD |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for g in data["groups"]:
        layers = ",".join(str(x) for x in (g["layers"] or [])) or "—"
        lines.append(
            f"| `{g['group_id']}` | {layers} | **{g['assigned']}** | "
            f"{_mb(g['bytes_baseline']):.2f} | {_mb(g['bytes_final']):.2f} | "
            f"{_mb(g['bytes_saved']):.2f} | {g['compress_pct']}% | "
            f"{g['delta_kld']:.4f} |"
        )

    lines += [
        "",
        "## What happened (plain language)",
        "",
    ]
    # Narrative bullets
    hist = data["quant_histogram"]
    heavily = [g for g in data["groups"] if g["assigned"] in {"Q2_K", "Q3_K"}]
    protected = [g for g in data["groups"] if g["assigned"] in {"Q5_K", "Q6_K", "Q8_0"}]
    lines.append(
        f"- Started from a **{data['baseline']}** baseline for size accounting, "
        f"then the optimizer downgraded groups with the best bytes/ΔKLD ratio."
    )
    lines.append(
        f"- **{len(heavily)}** groups landed at aggressive types (Q2/Q3); "
        f"**{len(protected)}** groups stayed protected (Q5+)."
    )
    for g in protected:
        lines.append(
            f"- Kept **{g['group_id']}** at **{g['assigned']}** "
            f"(compress {g['compress_pct']}%, ΔKLD≈{g['delta_kld']:.4f})."
        )
    # Top savers
    top = sorted(data["groups"], key=lambda x: x["bytes_saved"], reverse=True)[:5]
    lines.append("- Largest byte savers:")
    for g in top:
        lines.append(
            f"  - `{g['group_id']}` → {g['assigned']}: "
            f"saved {_mb(g['bytes_saved']):.2f} MiB ({g['compress_pct']}%)"
        )
    lines.append(
        f"- Non-quantizable norms/etc. remain F32 "
        f"({_mb(data['size']['fixed_nonquant_bytes']):.2f} MiB)."
    )
    lines += ["", "## Notes", ""]
    for n in data.get("notes") or []:
        lines.append(f"- {n}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report_card_html(data: dict[str, Any], path: Path) -> None:
    arch = data["architecture"]
    size = data["size"]
    esc = html_lib.escape

    def badge(q: str) -> str:
        colors = {
            "Q8_0": "#1b4332",
            "Q6_K": "#2d6a4f",
            "Q5_K": "#40916c",
            "Q4_K": "#b08968",
            "Q3_K": "#9c6644",
            "Q2_K": "#9b2226",
        }
        c = colors.get(q, "#333")
        return (
            f'<span style="background:{c};color:#fff;padding:2px 8px;'
            f'border-radius:999px;font-size:12px;font-weight:600">{esc(q)}</span>'
        )

    layer_rows = []
    for lc in data["layers"]:
        cells = [f"<td>{lc['layer']}</td><td>{esc(str(lc['depth']))}</td>"]
        for role in [
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_o",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
        ]:
            info = (lc.get("roles") or {}).get(role)
            if not info:
                cells.append("<td class='muted'>—</td>")
            else:
                cells.append(
                    f"<td>{badge(info['assigned'])}"
                    f"<div class='pct'>{info['compress_pct']}%</div></td>"
                )
        cells.append(f"<td><strong>{lc['compress_pct']}%</strong></td>")
        layer_rows.append("<tr>" + "".join(cells) + "</tr>")

    group_rows = []
    for g in data["groups"]:
        layers = ",".join(str(x) for x in (g["layers"] or [])) or "—"
        group_rows.append(
            "<tr>"
            f"<td><code>{esc(g['group_id'])}</code></td>"
            f"<td>{esc(layers)}</td>"
            f"<td>{badge(g['assigned'])}</td>"
            f"<td class='num'>{_mb(g['bytes_baseline']):.2f}</td>"
            f"<td class='num'>{_mb(g['bytes_final']):.2f}</td>"
            f"<td class='num'>{_mb(g['bytes_saved']):.2f}</td>"
            f"<td class='num'>{g['compress_pct']}%</td>"
            f"<td class='num'>{g['delta_kld']:.4f}</td>"
            "</tr>"
        )

    role_rows = []
    for r in data["role_summary"]:
        role_rows.append(
            "<tr>"
            f"<td>{esc(r['role'])}</td>"
            f"<td class='num'>{r['groups']}</td>"
            f"<td class='num'>{_mb(r['bytes_baseline']):.2f}</td>"
            f"<td class='num'>{_mb(r['bytes_final']):.2f}</td>"
            f"<td class='num'>{_mb(r['bytes_saved']):.2f}</td>"
            f"<td class='num'>{r['compress_pct']}%</td>"
            f"<td><code>{esc(json.dumps(r['assignments']))}</code></td>"
            "</tr>"
        )

    hist = " · ".join(
        f"{badge(k)} ×{v}" for k, v in (data.get("quant_histogram") or {}).items()
    )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Quantization Report Card — {esc(data['model_ref'])}</title>
<style>
  :root {{
    --bg: #f6f1ea;
    --ink: #1c1917;
    --card: #fffdf8;
    --line: #e7e0d5;
    --accent: #0f766e;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 2rem 1.25rem 4rem;
    font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
    background:
      radial-gradient(1200px 600px at 10% -10%, #d8efe9 0%, transparent 55%),
      radial-gradient(900px 500px at 100% 0%, #f0e2c8 0%, transparent 50%),
      var(--bg);
    color: var(--ink);
  }}
  .wrap {{ max-width: 1100px; margin: 0 auto; }}
  h1 {{ font-size: 2.1rem; margin: 0 0 .25rem; letter-spacing: -0.02em; }}
  h2 {{ margin-top: 2.2rem; font-size: 1.35rem; border-bottom: 2px solid var(--ink); padding-bottom: .35rem; }}
  .sub {{ color: #57534e; margin-bottom: 1.5rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: .75rem; }}
  @media (max-width: 800px) {{ .grid {{ grid-template-columns: 1fr 1fr; }} }}
  .stat {{
    background: var(--card); border: 1px solid var(--line); border-radius: 14px;
    padding: 1rem 1.1rem; box-shadow: 0 1px 0 rgba(0,0,0,.04);
  }}
  .stat .label {{ font-size: .75rem; text-transform: uppercase; letter-spacing: .06em; color: #78716c; }}
  .stat .value {{ font-size: 1.55rem; font-weight: 700; margin-top: .2rem; }}
  .verdict {{
    display: inline-block; margin-top: .5rem; padding: .35rem .8rem;
    border-radius: 999px; background: var(--accent); color: white; font-weight: 700;
  }}
  table {{
    width: 100%; border-collapse: collapse; background: var(--card);
    border: 1px solid var(--line); border-radius: 12px; overflow: hidden;
    font-family: ui-sans-serif, system-ui, sans-serif; font-size: 13px;
  }}
  th, td {{ padding: .55rem .6rem; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
  th {{ background: #efe8dc; font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }}
  tr:last-child td {{ border-bottom: none; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .pct {{ color: #57534e; font-size: 11px; margin-top: 2px; }}
  .muted {{ color: #a8a29e; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }}
  .hist {{ margin: .75rem 0 0; line-height: 1.9; }}
  .note {{ color: #57534e; font-size: .95rem; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Quantization Report Card</h1>
  <p class="sub">
    Model <code>{esc(data['model_ref'])}</code> ·
    {esc(str(arch.get('family') or '—'))} ·
    {arch.get('n_layers')} layers ·
    {arch.get('n_tensors')} tensors ·
    {arch.get('n_quantizable')} quantizable
  </p>
  <div class="verdict">{esc(str(data['quality'].get('verdict') or 'n/a'))}</div>

  <h2>Size at a glance</h2>
  <div class="grid">
    <div class="stat"><div class="label">Baseline ({esc(data['baseline'])})</div>
      <div class="value">{size['baseline_total_mb']:.1f}<span style="font-size:.9rem"> MiB</span></div></div>
    <div class="stat"><div class="label">After recipe</div>
      <div class="value">{size['estimated_total_mb']:.1f}<span style="font-size:.9rem"> MiB</span></div></div>
    <div class="stat"><div class="label">Compression</div>
      <div class="value">{size['quantizable_compress_pct']}%</div></div>
    <div class="stat"><div class="label">Pred. Σ ΔKLD</div>
      <div class="value">{data['quality'].get('predicted_delta_kld_sum', 0):.3f}</div></div>
  </div>
  <p class="hist">Quant mix: {hist}</p>

  <h2>Compression by role</h2>
  <table>
    <thead><tr>
      <th>Role</th><th class="num">Groups</th><th class="num">Baseline MiB</th>
      <th class="num">Final MiB</th><th class="num">Saved</th><th class="num">Compress</th><th>Assignments</th>
    </tr></thead>
    <tbody>
      {''.join(role_rows)}
    </tbody>
  </table>

  <h2>Per-layer report card</h2>
  <p class="note">Each cell shows the assigned quant type and compression % vs {esc(data['baseline'])} for that layer’s share of the group.</p>
  <table>
    <thead><tr>
      <th>Layer</th><th>Depth</th>
      <th>attn_q</th><th>attn_k</th><th>attn_v</th><th>attn_o</th>
      <th>ffn_gate</th><th>ffn_up</th><th>ffn_down</th>
      <th>Layer</th>
    </tr></thead>
    <tbody>
      {''.join(layer_rows)}
    </tbody>
  </table>

  <h2>Per-group detail</h2>
  <table>
    <thead><tr>
      <th>Group</th><th>Layers</th><th>Assigned</th>
      <th class="num">Base MiB</th><th class="num">Final MiB</th>
      <th class="num">Saved</th><th class="num">Compress</th><th class="num">ΔKLD</th>
    </tr></thead>
    <tbody>
      {''.join(group_rows)}
    </tbody>
  </table>

  <h2>Notes</h2>
  <ul class="note">
    {''.join(f'<li>{esc(n)}</li>' for n in (data.get('notes') or []))}
  </ul>
</div>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def write_quantization_report_card(
    out_dir: Path,
    *,
    model_ref: str,
    catalog: dict[str, Any],
    assignments: dict[str, str],
    sensitivity_rows: list[dict[str, Any]] | None = None,
    optimize_manifest: dict[str, Any] | None = None,
    validate_payload: dict[str, Any] | None = None,
    resolve_descriptor: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write html/md/json report card. Returns paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = build_report_card_data(
        model_ref=model_ref,
        catalog=catalog,
        assignments=assignments,
        sensitivity_rows=sensitivity_rows,
        optimize_manifest=optimize_manifest,
        validate_payload=validate_payload,
        resolve_descriptor=resolve_descriptor,
    )
    json_path = out_dir / "quantization_report_card.json"
    md_path = out_dir / "quantization_report_card.md"
    html_path = out_dir / "quantization_report_card.html"
    json_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    write_report_card_md(data, md_path)
    write_report_card_html(data, html_path)
    return {
        "json": str(json_path),
        "md": str(md_path),
        "html": str(html_path),
    }
