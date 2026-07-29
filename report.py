"""HTML report generator — feature 03 of the platform (docs/platform/03-…).

Read-only over the run store: renders artifacts that steps already wrote into
one self-contained ``report.html`` (inline CSS + SVG, no external assets, works
offline from ``file://``). Missing artifacts render as "not run" — never an
error. Every section captions the artifact it came from.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sensitivity import estimate_group_nbytes

# Role pins mirrored from optimizer.DEFAULT_PINS (import kept light on purpose)
_PIN_ROLES = {"embedding": "Q8_0", "lm_head": "Q8_0", "attn_v": "Q5_K"}

_QUANT_COLORS = {
    "Q8_0": "#4dd0a5",
    "Q6_K": "#59b8e6",
    "Q5_K": "#7f96f0",
    "Q4_K": "#b58cf0",
    "Q3_K": "#e08cd0",
    "Q2_K": "#ef8080",
}


def _esc(s: Any) -> str:
    return html.escape(str(s))


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _mb(nbytes: float) -> str:
    return f"{nbytes / (1024**2):.1f} MB"


# ---------------------------------------------------------------------------
# Data extraction (one function per section; all pure reads)
# ---------------------------------------------------------------------------


def _steps_dir(run_root: Path) -> Path:
    return run_root / "steps"


def extract_summary(run_root: Path) -> dict[str, Any]:
    run_meta = _read_json(run_root / "run.json") or {}
    resolve = _read_json(_steps_dir(run_root) / "01_resolve" / "output.json")
    fit_plan = _read_json(run_root / "fit_plan.json")
    if not run_meta and not resolve:
        return {"available": False}
    desc = (resolve or {}).get("descriptor") or {}
    return {
        "available": True,
        "source": "run.json · steps/01_resolve/output.json",
        "model_ref": run_meta.get("model_ref") or (resolve or {}).get("user_ref"),
        "run_id": run_meta.get("run_id"),
        "status": run_meta.get("status"),
        "quant": run_meta.get("quant_label") or run_meta.get("quant_format"),
        "hf_repo_id": (resolve or {}).get("hf_repo_id"),
        "family": desc.get("family"),
        "layers": desc.get("layer_count"),
        "params": desc.get("parameter_count"),
        "context_length": desc.get("context_length"),
        "specialty_domain": desc.get("specialty_domain"),
        "fit_plan": fit_plan,
    }


def _load_catalog(run_root: Path) -> dict[str, Any] | None:
    for step in ("08_activation_features", "06_weight_features", "05_catalog"):
        cat = _read_json(_steps_dir(run_root) / step / "tensor_catalog.json")
        if cat:
            return cat
    return None


def _load_sensitivity(run_root: Path) -> dict[str, Any] | None:
    return _read_json(_steps_dir(run_root) / "12_sensitivity" / "sensitivity.json")


def extract_sensitivity(run_root: Path) -> dict[str, Any]:
    sens = _load_sensitivity(run_root)
    if not sens or not sens.get("rows"):
        return {"available": False}
    rows = sens["rows"]
    groups = sorted({r["group_id"] for r in rows})
    probes = list(sens.get("probe_types") or sorted({r["probe"] for r in rows}))
    cells: dict[str, dict[str, float]] = {}
    for r in rows:
        cells.setdefault(r["group_id"], {})[str(r["probe"]).upper()] = float(
            r.get("delta_kld") or 0.0
        )
    return {
        "available": True,
        "source": "steps/12_sensitivity/sensitivity.json",
        "method": sens.get("method"),
        "baseline": sens.get("baseline_type"),
        "groups": groups,
        "probes": [p.upper() for p in probes],
        "cells": cells,
    }


def extract_allocations(run_root: Path) -> dict[str, Any]:
    opt = _read_json(_steps_dir(run_root) / "13_optimize" / "output.json")
    if not opt or not opt.get("assignments"):
        return {"available": False}
    manifest = _read_json(_steps_dir(run_root) / "13_optimize" / "optimize_manifest.json") or {}
    catalog = _load_catalog(run_root) or {}
    sens = _load_sensitivity(run_root) or {}

    row_index = {
        (r["group_id"], str(r["probe"]).upper()): r for r in sens.get("rows") or []
    }
    baseline = str(sens.get("baseline_type") or "Q6_K").upper()
    history = (manifest.get("primary") or {}).get("history") or []
    last_move: dict[str, dict[str, Any]] = {}
    for h in history:
        last_move[h["group_id"]] = h

    groups_meta = catalog.get("groups") or {}
    tensors = catalog.get("tensors") or {}

    def group_elements(gid: str) -> int:
        g = groups_meta.get(gid) or {}
        return sum(
            int((tensors.get(n) or {}).get("n_elements") or 0)
            for n in g.get("tensor_names") or []
        )

    rows: list[dict[str, Any]] = []
    for gid, q in sorted(opt["assignments"].items()):
        q = str(q).upper()
        role = gid.split("@")[0]
        n_elem = group_elements(gid)
        try:
            nbytes = estimate_group_nbytes(n_elem, q) if n_elem else None
        except KeyError:
            nbytes = None
        srow = row_index.get((gid, q))
        dkld = float(srow["delta_kld"]) if srow else (0.0 if q == baseline else None)

        if role in _PIN_ROLES and q == _PIN_ROLES[role].upper():
            reason = f"pinned: role floor {_PIN_ROLES[role]}"
        elif q == baseline:
            reason = f"held at baseline {baseline} — no downgrade paid off"
        elif gid in last_move:
            mv = last_move[gid]
            reason = (
                f"greedy downgrade → {mv['to']}: saved {_mb(mv['delta_bytes'])} "
                f"for ΔKLD +{mv['delta_kld_inc']:.4f}"
            )
        else:
            reason = "assigned by optimizer"
        rows.append(
            {
                "group": gid,
                "quant": q,
                "bytes": nbytes,
                "delta_kld": dkld,
                "reason": reason,
            }
        )

    return {
        "available": True,
        "source": "steps/13_optimize/{output.json, optimize_manifest.json}",
        "method": opt.get("method"),
        "budget_bytes": opt.get("budget_bytes"),
        "estimated_bytes": opt.get("estimated_bytes"),
        "predicted_delta_kld": opt.get("predicted_delta_kld"),
        "baseline": baseline,
        "rows": rows,
        "n_decisions": len(history),
    }


def extract_pareto(run_root: Path) -> dict[str, Any]:
    manifest = _read_json(_steps_dir(run_root) / "13_optimize" / "optimize_manifest.json")
    if not manifest or not manifest.get("pareto"):
        return {"available": False}
    chosen_bytes = (manifest.get("primary") or {}).get("estimated_bytes")
    points = [
        {
            "size_mb": p["estimated_bytes"] / (1024**2),
            "kld": p["predicted_delta_kld"],
            "chosen": p["estimated_bytes"] == chosen_bytes,
            "meets_budget": p.get("meets_budget"),
        }
        for p in manifest["pareto"]
    ]
    return {
        "available": True,
        "source": "steps/13_optimize/optimize_manifest.json",
        "points": points,
    }


def extract_gates(run_root: Path) -> dict[str, Any]:
    val = _read_json(_steps_dir(run_root) / "15_validate" / "output.json")
    if not val:
        return {"available": False}
    tier1 = val.get("tier1") or {}
    metrics = tier1.get("metrics") or {}
    thresholds = tier1.get("gates") or {}
    gate_rows = []
    for key, thr_key, higher_better in (
        ("mean_kld", "mean_kld_max", False),
        ("p999_kld", "p999_kld_max", False),
        ("max_kld", "max_kld_max", False),
        ("top1_agree", "top1_agree_min", True),
    ):
        value = metrics.get(key)
        thr = thresholds.get(thr_key)
        if value is None or thr is None:
            continue
        ok = value >= thr if higher_better else value <= thr
        gate_rows.append(
            {
                "metric": key,
                "value": float(value),
                "threshold": float(thr),
                "higher_better": higher_better,
                "pass": bool(ok),
            }
        )
    return {
        "available": True,
        "source": "steps/15_validate/output.json",
        "verdict": val.get("verdict"),
        "method": val.get("method"),
        "tier1_pass": tier1.get("pass"),
        "tier1_note": tier1.get("note"),
        "gates": gate_rows,
    }


def extract_benchmarks(run_root: Path) -> dict[str, Any]:
    from benchmark import find_run_benchresults

    results = find_run_benchresults(run_root)
    if not results:
        return {"available": False}
    return {
        "available": True,
        "source": "benchmarks/*/benchresult.json",
        "results": results,
    }


def extract_reproducibility(run_root: Path) -> dict[str, Any]:
    resolve = _read_json(_steps_dir(run_root) / "01_resolve" / "output.json") or {}
    freeze = _read_json(_steps_dir(run_root) / "09_freeze_gguf" / "output.json") or {}
    imatrix = _read_json(_steps_dir(run_root) / "10_imatrix" / "output.json") or {}
    corpus = _read_json(_steps_dir(run_root) / "07_corpus" / "output.json") or {}
    opt = _read_json(_steps_dir(run_root) / "13_optimize" / "output.json") or {}
    rows = [
        ("model source", resolve.get("hf_repo_id") or resolve.get("user_ref")),
        ("source sha256", resolve.get("source_sha256")),
        ("frozen gguf sha256", freeze.get("gguf_sha256")),
        ("imatrix sha256", imatrix.get("imatrix_sha256")),
        ("corpus id", corpus.get("corpus_id")),
        ("recipe", opt.get("recipe_path")),
    ]
    rows = [(k, v) for k, v in rows if v]
    if not rows:
        return {"available": False}
    return {
        "available": True,
        "source": "steps/{01_resolve,07_corpus,09_freeze_gguf,10_imatrix,13_optimize}/output.json",
        "rows": rows,
    }


def build_report_data(run_root: Path) -> dict[str, Any]:
    run_root = Path(run_root)
    return {
        "summary": extract_summary(run_root),
        "allocations": extract_allocations(run_root),
        "sensitivity": extract_sensitivity(run_root),
        "pareto": extract_pareto(run_root),
        "gates": extract_gates(run_root),
        "benchmarks": extract_benchmarks(run_root),
        "reproducibility": extract_reproducibility(run_root),
    }


# ---------------------------------------------------------------------------
# SVG / HTML rendering (no external assets)
# ---------------------------------------------------------------------------


def _svg_hbar(items: list[tuple[str, float, str, str]], *, width: int = 860) -> str:
    """items: (label, value, color, annotation). Bars scaled to max value."""
    if not items:
        return ""
    row_h, gap, label_w, annot_w = 22, 6, 190, 150
    bar_w = width - label_w - annot_w - 20
    vmax = max(v for _, v, _, _ in items) or 1.0
    h = len(items) * (row_h + gap)
    parts = [
        f'<svg viewBox="0 0 {width} {h}" width="100%" '
        f'xmlns="http://www.w3.org/2000/svg" role="img">'
    ]
    for i, (label, value, color, annot) in enumerate(items):
        y = i * (row_h + gap)
        w = max(2.0, bar_w * value / vmax)
        parts.append(
            f'<text x="{label_w - 8}" y="{y + row_h - 6}" text-anchor="end" '
            f'class="svgt">{_esc(label)}</text>'
            f'<rect x="{label_w}" y="{y + 3}" width="{w:.1f}" height="{row_h - 6}" '
            f'rx="3" fill="{color}"/>'
            f'<text x="{label_w + w + 8:.1f}" y="{y + row_h - 6}" class="svgt dim">'
            f"{_esc(annot)}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _kld_color(v: float, vmax: float) -> str:
    """0 → deep green, vmax → red, log-ish ramp in between."""
    if vmax <= 0:
        t = 0.0
    else:
        t = min(1.0, max(0.0, v / vmax)) ** 0.5
    r = int(46 + t * (214 - 46))
    g = int(160 - t * (160 - 74))
    b = int(122 - t * (122 - 74))
    return f"rgb({r},{g},{b})"


def _heatmap_table(sens: dict[str, Any]) -> str:
    groups, probes, cells = sens["groups"], sens["probes"], sens["cells"]
    vmax = max(
        (v for g in cells.values() for v in g.values()), default=0.0
    )
    head = "".join(f"<th>{_esc(p)}</th>" for p in probes)
    body = []
    for gid in groups:
        tds = []
        for p in probes:
            v = cells.get(gid, {}).get(p)
            if v is None:
                tds.append('<td class="dim">·</td>')
            else:
                tds.append(
                    f'<td style="background:{_kld_color(v, vmax)};color:#0b1220">'
                    f"{v:.4f}</td>"
                )
        body.append(f"<tr><th>{_esc(gid)}</th>{''.join(tds)}</tr>")
    return (
        '<table class="heat"><thead><tr><th>group ↓ / probe →</th>'
        f"{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"
        f'<p class="cap">ΔKLD vs {_esc(sens.get("baseline") or "baseline")} — '
        "green: cheap to compress · red: expensive.</p>"
    )


def _svg_scatter(points: list[dict[str, Any]], *, width: int = 640, height: int = 320) -> str:
    if not points:
        return ""
    pad = 52
    xs = [p["size_mb"] for p in points]
    ys = [p["kld"] for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    xr = (x1 - x0) or 1.0
    yr = (y1 - y0) or 1.0

    def sx(v: float) -> float:
        return pad + (v - x0) / xr * (width - 2 * pad)

    def sy(v: float) -> float:
        return height - pad - (v - y0) / yr * (height - 2 * pad)

    ordered = sorted(points, key=lambda p: p["size_mb"])
    path = " ".join(
        f"{'M' if i == 0 else 'L'}{sx(p['size_mb']):.1f},{sy(p['kld']):.1f}"
        for i, p in enumerate(ordered)
    )
    dots = []
    for p in ordered:
        chosen = p.get("chosen")
        dots.append(
            f'<circle cx="{sx(p["size_mb"]):.1f}" cy="{sy(p["kld"]):.1f}" '
            f'r="{9 if chosen else 5}" fill="{"#f2c14e" if chosen else "#59b8e6"}" '
            f'stroke="#0b1220" stroke-width="1.5"/>'
        )
        if chosen:
            dots.append(
                f'<text x="{sx(p["size_mb"]) + 12:.1f}" y="{sy(p["kld"]) + 4:.1f}" '
                f'class="svgt">chosen · {p["size_mb"]:.0f} MB</text>'
            )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" '
        f'xmlns="http://www.w3.org/2000/svg" role="img">'
        f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" class="ax"/>'
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" class="ax"/>'
        f'<text x="{width / 2:.0f}" y="{height - 12}" text-anchor="middle" class="svgt dim">size (MB)</text>'
        f'<text x="16" y="{height / 2:.0f}" transform="rotate(-90 16 {height / 2:.0f})" '
        f'text-anchor="middle" class="svgt dim">predicted ΔKLD</text>'
        f'<path d="{path}" fill="none" stroke="#3a4a63" stroke-width="1.5" stroke-dasharray="4 3"/>'
        f"{''.join(dots)}</svg>"
    )


def _section(title: str, source: str | None, inner: str) -> str:
    cap = f'<p class="cap src">source: {_esc(source)}</p>' if source else ""
    return f'<section><h2>{_esc(title)}</h2>{inner}{cap}</section>'


def _not_run(title: str, hint: str) -> str:
    return (
        f"<section><h2>{_esc(title)}</h2>"
        f'<p class="dim">not run — {_esc(hint)}</p></section>'
    )


_CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; padding: 32px 20px 80px; background: #0b1220; color: #dbe4f0;
  font: 15px/1.55 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
main { max-width: 960px; margin: 0 auto; }
h1 { font-size: 26px; margin: 0 0 4px; }
h1 span { color: #59d6e6; }
h2 { font-size: 18px; margin: 0 0 12px; color: #9fd8e8;
  border-bottom: 1px solid #22314a; padding-bottom: 6px; }
section { background: #111a2c; border: 1px solid #22314a; border-radius: 12px;
  padding: 20px 22px; margin: 18px 0; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; }
th, td { padding: 5px 9px; text-align: left; }
thead th { color: #8aa2c0; font-weight: 600; border-bottom: 1px solid #2a3a55; }
tbody tr:nth-child(odd) { background: #0e1626; }
tbody th { font-weight: 500; color: #c6d4e6; white-space: nowrap; }
.heat td { text-align: center; font-variant-numeric: tabular-nums; border-radius: 3px; }
.badge { display: inline-block; padding: 2px 12px; border-radius: 999px;
  font-weight: 700; font-size: 13px; }
.badge.RELEASE { background: #1d7a4f; color: #fff; }
.badge.PROVISIONAL { background: #b98a1c; color: #14100a; }
.badge.FAIL { background: #a33; color: #fff; }
.pass { color: #4dd0a5; font-weight: 600; } .fail { color: #ef8080; font-weight: 600; }
.dim, .cap { color: #71829c; } .cap { font-size: 12px; margin: 10px 0 0; }
.cap.src { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.svgt { font: 12px -apple-system, sans-serif; fill: #c6d4e6; }
.svgt.dim { fill: #71829c; }
.ax { stroke: #2a3a55; stroke-width: 1; }
.qtag { display: inline-block; min-width: 46px; text-align: center; padding: 1px 7px;
  border-radius: 5px; font-weight: 700; font-size: 12px; color: #0b1220; }
.kv { display: grid; grid-template-columns: 180px 1fr; gap: 3px 14px; font-size: 14px; }
.kv b { color: #8aa2c0; font-weight: 600; }
code { background: #0e1626; border: 1px solid #22314a; border-radius: 5px;
  padding: 1px 6px; font-size: 12.5px; }
"""


def render_html(data: dict[str, Any]) -> str:
    parts: list[str] = []
    s = data["summary"]

    # --- header + summary ---
    title_model = s.get("model_ref") or "unknown model" if s.get("available") else "unknown model"
    if s.get("available"):
        kv_rows = [
            ("model", s.get("model_ref")),
            ("run", s.get("run_id")),
            ("status", s.get("status")),
            ("quant target", s.get("quant")),
            ("hf source", s.get("hf_repo_id")),
            ("family", s.get("family")),
            ("layers", s.get("layers")),
            ("parameters", f"{s['params']:,}" if s.get("params") else None),
            ("trained context", s.get("context_length")),
            ("specialty domain", s.get("specialty_domain")),
        ]
        kv_html = "".join(
            f"<b>{_esc(k)}</b><span>{_esc(v)}</span>" for k, v in kv_rows if v is not None
        )
        inner = f'<div class="kv">{kv_html}</div>'
        fit = s.get("fit_plan")
        if fit:
            frows = [
                ("hardware", (fit.get("profile") or {}).get("id")),
                ("context sized for", fit.get("ctx")),
                ("usable memory", f"{fit.get('usable_gb', 0):.2f} GB"),
                ("kv cache", f"−{fit.get('kv_cache_gb', 0):.2f} GB"),
                (
                    "weight budget",
                    f"{fit.get('weight_budget_bytes', 0) / (1024**3):.2f} GB",
                ),
            ]
            fkv = "".join(
                f"<b>{_esc(k)}</b><span>{_esc(v)}</span>" for k, v in frows if v is not None
            )
            inner += (
                '<h2 style="margin-top:18px">Hardware fit</h2>'
                f'<div class="kv">{fkv}</div>'
            )
        parts.append(_section("Model", s.get("source"), inner))
    else:
        parts.append(_not_run("Model", "no run.json / resolve output found"))

    # --- allocations ---
    a = data["allocations"]
    if a.get("available"):
        head = (
            "<thead><tr><th>group</th><th>assigned</th><th>size</th>"
            "<th>ΔKLD</th><th>why</th></tr></thead>"
        )
        rows_html = []
        for r in a["rows"]:
            color = _QUANT_COLORS.get(r["quant"], "#8aa2c0")
            dkld = f"+{r['delta_kld']:.4f}" if r["delta_kld"] else ("0" if r["delta_kld"] == 0.0 else "—")
            size = _mb(r["bytes"]) if r["bytes"] else "—"
            rows_html.append(
                f"<tr><th>{_esc(r['group'])}</th>"
                f'<td><span class="qtag" style="background:{color}">{_esc(r["quant"])}</span></td>'
                f"<td>{size}</td><td>{dkld}</td>"
                f'<td class="dim">{_esc(r["reason"])}</td></tr>'
            )
        meta = (
            f'<p>budget {_mb(a["budget_bytes"])} · estimated {_mb(a["estimated_bytes"])} · '
            f'predicted ΔKLD +{a["predicted_delta_kld"]:.4f} · '
            f'{a["n_decisions"]} greedy decisions ({_esc(a["method"])})</p>'
        )
        table = f"<table>{head}<tbody>{''.join(rows_html)}</tbody></table>"

        bar_items = [
            (r["group"], float(r["bytes"]), _QUANT_COLORS.get(r["quant"], "#8aa2c0"), _mb(r["bytes"]))
            for r in sorted(a["rows"], key=lambda x: -(x["bytes"] or 0))
            if r["bytes"]
        ]
        bytes_chart = _svg_hbar(bar_items) if bar_items else ""
        parts.append(_section("Bit allocation", a.get("source"), meta + table))
        if bytes_chart:
            parts.append(
                _section(
                    "Where the bytes live",
                    a.get("source"),
                    bytes_chart
                    + '<p class="cap">Estimated group size at its assigned quant type.</p>',
                )
            )
    else:
        parts.append(_not_run("Bit allocation", "step 13 (optimize) has not produced a recipe"))

    # --- sensitivity heatmap ---
    sens = data["sensitivity"]
    if sens.get("available"):
        note = ""
        if "proxy" in str(sens.get("method") or ""):
            note = (
                '<p class="cap">⚠ proxy estimates from features — not measured '
                "probes. Run step 12 with llama.cpp tools for measured ΔKLD.</p>"
            )
        parts.append(
            _section("Sensitivity heatmap", sens.get("source"), _heatmap_table(sens) + note)
        )
    else:
        parts.append(_not_run("Sensitivity heatmap", "step 12 (sensitivity) has not run"))

    # --- pareto ---
    p = data["pareto"]
    if p.get("available"):
        parts.append(
            _section(
                "Size ↔ quality frontier",
                p.get("source"),
                _svg_scatter(p["points"])
                + '<p class="cap">Each point is a full recipe optimized at a '
                "different budget; the highlighted point is the shipped one.</p>",
            )
        )
    else:
        parts.append(_not_run("Size ↔ quality frontier", "no Pareto set from step 13"))

    # --- gates ---
    g = data["gates"]
    if g.get("available"):
        verdict = str(g.get("verdict") or "?")
        inner = f'<p>verdict <span class="badge {_esc(verdict)}">{_esc(verdict)}</span></p>'
        if g["gates"]:
            head = (
                "<thead><tr><th>metric</th><th>value</th><th>gate</th>"
                "<th>result</th></tr></thead>"
            )
            rws = []
            for row in g["gates"]:
                op = "≥" if row["higher_better"] else "≤"
                cls = "pass" if row["pass"] else "fail"
                word = "pass" if row["pass"] else "fail"
                rws.append(
                    f"<tr><th>{_esc(row['metric'])}</th><td>{row['value']:.6g}</td>"
                    f"<td>{op} {row['threshold']:.6g}</td>"
                    f'<td class="{cls}">{word}</td></tr>'
                )
            inner += f"<table>{head}<tbody>{''.join(rws)}</tbody></table>"
        if g.get("tier1_note"):
            inner += f'<p class="cap">{_esc(g["tier1_note"])}</p>'
        parts.append(_section("Validation gates (Tier 1)", g.get("source"), inner))
    else:
        parts.append(_not_run("Validation gates", "step 15 (validate) has not run"))

    # --- benchmarks ---
    b = data["benchmarks"]
    if b.get("available"):
        blocks = []
        for res in b["results"]:
            tasks = (res.get("quality") or {}).get("tasks") or {}
            tp = res.get("throughput") or {}
            rws = []
            for tid, t in tasks.items():
                delta = (
                    f"{t['paired_delta']:+.4f} [{t['ci_low']:+.4f}, {t['ci_high']:+.4f}]"
                    if "paired_delta" in t
                    else '<span class="dim">no reference</span>'
                )
                score = f"{t['score']:.4f}" if t.get("score") is not None else "—"
                rws.append(
                    f"<tr><th>{_esc(tid)}</th><td>{score}</td><td>{delta}</td></tr>"
                )
            tbl = (
                "<table><thead><tr><th>task</th><th>score</th>"
                "<th>Δ vs BF16 (95% CI)</th></tr></thead>"
                f"<tbody>{''.join(rws)}</tbody></table>"
                if rws
                else '<p class="dim">quality tasks skipped '
                f"({_esc((res.get('quality') or {}).get('reason') or 'n/a')})</p>"
            )
            tp_line = (
                f"pp {tp.get('pp_tps', '—')} t/s · tg {tp.get('tg_tps', '—')} t/s"
                + (f" on {_esc(tp['device'])}" if tp.get("device") else "")
                if tp
                else "throughput not measured"
            )
            blocks.append(
                f"<p><b>suite {_esc(res.get('suite'))}</b> · {_esc(res.get('created_at'))} · "
                f"{tp_line} · gguf <code>{_esc((res.get('gguf_sha256') or '')[:16])}…</code></p>"
                + tbl
            )
        parts.append(_section("Benchmarks", b.get("source"), "".join(blocks)))
    else:
        parts.append(
            _not_run("Benchmarks", "no benchresult.json — run: odg benchmark <gguf> --suite smoke")
        )

    # --- reproducibility ---
    r = data["reproducibility"]
    if r.get("available"):
        kvh = "".join(
            f"<b>{_esc(k)}</b><span><code>{_esc(v)}</code></span>" for k, v in r["rows"]
        )
        parts.append(
            _section(
                "Reproducibility",
                r.get("source"),
                f'<div class="kv">{kvh}</div>'
                '<p class="cap">The recipe + these hashes rebuild the GGUF '
                "bit-for-bit: <code>odg export</code> from the same inputs.</p>",
            )
        )
    else:
        parts.append(_not_run("Reproducibility", "no provenance artifacts found"))

    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>OpenDynamicGGUF report — {_esc(title_model)}</title>"
        f"<style>{_CSS}</style></head><body><main>"
        f"<h1><span>OpenDynamicGGUF</span> optimization report</h1>"
        f'<p class="dim">{_esc(title_model)} — every number below links back to a '
        "run artifact; nothing is recomputed at render time.</p>"
        f"{''.join(parts)}"
        "</main></body></html>"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@dataclass
class ReportResult:
    run_root: str
    report_path: str
    sections_rendered: int
    sections_missing: list[str] = field(default_factory=list)
    steps_log: list[str] = field(default_factory=list)

    def summary_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def generate_report(run_root: Path, out_path: Path | None = None) -> ReportResult:
    run_root = Path(run_root)
    if not (run_root / "run.json").is_file():
        raise FileNotFoundError(f"Not a run directory (no run.json): {run_root}")

    data = build_report_data(run_root)
    html_text = render_html(data)
    out = Path(out_path) if out_path else run_root / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_text, encoding="utf-8")

    missing = [k for k, v in data.items() if not v.get("available")]
    rendered = len(data) - len(missing)
    log = [
        f"1. Extracted {len(data)} sections from {run_root}",
        f"2. Rendered {rendered} sections; missing: {missing or 'none'}",
        f"3. Wrote {out} ({out.stat().st_size / 1024:.0f} KB, self-contained)",
    ]
    return ReportResult(
        run_root=str(run_root),
        report_path=str(out),
        sections_rendered=rendered,
        sections_missing=missing,
        steps_log=log,
    )
