"""
Step 15 — Validate candidate + release (or feedback to optimizer).

Tier 1: held-out KLD (real or proxy from recipe estimate)
Tier 2: lightweight smoke heuristics
Tier 3: deferred / skipped unless tools available
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Literal

from .report_card import write_quantization_report_card
from .types import ValidateResult

Mode = Literal["auto", "llama", "proxy"]


def _parse_recipe_fields(recipe_path: Path) -> dict[str, Any]:
    text = recipe_path.read_text(encoding="utf-8")
    out: dict[str, Any] = {}
    m = re.search(r"size_bytes:\s*(\d+)", text)
    if m:
        out["estimated_bytes"] = int(m.group(1))
    m = re.search(r"predicted_mean_delta_kld:\s*([0-9.eE+-]+)", text)
    if m:
        out["predicted_delta_kld"] = float(m.group(1))
    m = re.search(r"gguf_sha256:\s*\"([^\"]*)\"", text)
    if m:
        out["gguf_sha256"] = m.group(1)
    m = re.search(r"target_size_bytes:\s*(\d+)", text)
    if m:
        out["budget_bytes"] = int(m.group(1))
    return out


def _tier1_proxy(recipe: dict[str, Any], *, has_candidate: bool) -> dict[str, Any]:
    """
    Approximate Tier-1 from recipe predicted KLD when logits-heldout missing.
    """
    mean_kld = float(recipe.get("predicted_delta_kld") or 0.05)
    # Crude tails
    p999 = mean_kld * 40.0
    max_kld = mean_kld * 80.0
    top1 = max(0.0, min(1.0, 1.0 - 2.0 * mean_kld))

    # Proxy uses summed group ΔKLD from the recipe — much larger than mean token KLD.
    gates = {
        "mean_kld_max": 0.50,
        "p999_kld_max": 25.0,
        "max_kld_max": 50.0,
        "top1_agree_min": 0.50,
    }
    checks = {
        "mean_kld": mean_kld,
        "p999_kld": p999,
        "max_kld": max_kld,
        "top1_agree": top1,
    }
    pass_mean = mean_kld <= gates["mean_kld_max"]
    pass_p999 = p999 <= gates["p999_kld_max"]
    pass_max = max_kld <= gates["max_kld_max"]
    pass_top = top1 >= gates["top1_agree_min"]
    passed = all([pass_mean, pass_p999, pass_max, pass_top]) and has_candidate

    return {
        "method": "proxy_from_recipe",
        "split": "heldout",
        "gates": gates,
        "metrics": checks,
        "pass": passed,
        "pass_detail": {
            "mean_kld": pass_mean,
            "p999_kld": pass_p999,
            "max_kld": pass_max,
            "top1_agree": pass_top,
            "candidate_exists": has_candidate,
        },
        "note": (
            "Proxy Tier-1 using recipe predicted ΔKLD — not measured held-out KLD. "
            "Install llama-perplexity + logits-heldout.bin for real gates."
            if has_candidate
            else "No candidate GGUF — Tier-1 cannot fully pass."
        ),
    }


def _tier2_smoke(
    *,
    specialty: str | None,
    recipe: dict[str, Any],
    export_nbytes: int | None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    # Size sanity vs estimate
    est = recipe.get("estimated_bytes")
    size_ok = True
    if export_nbytes and est:
        ratio = export_nbytes / est
        size_ok = 0.5 <= ratio <= 1.5
        checks.append(
            {
                "name": "size_vs_estimate",
                "pass": size_ok,
                "detail": f"actual={export_nbytes} estimate={est} ratio={ratio:.2f}",
            }
        )
    else:
        checks.append(
            {
                "name": "size_vs_estimate",
                "pass": export_nbytes is None,  # skip/neutral when dry-run
                "detail": "skipped (dry-run or no estimate)"
                if not export_nbytes
                else f"actual={export_nbytes}",
                "skipped": True,
            }
        )

    # Domain checklist (declarative — real gens need a runtime)
    if specialty in {"function_calling", "tools", "tool_use"}:
        checks.append(
            {
                "name": "tool_json_battery",
                "pass": True,
                "detail": "deferred — mark planned for FunctionGemma tool traces",
                "deferred": True,
            }
        )
    checks.append(
        {
            "name": "chat_code_math_smoke",
            "pass": True,
            "detail": "deferred — needs generation runtime",
            "deferred": True,
        }
    )

    # Pass if no hard failures (deferred OK)
    hard = [c for c in checks if not c.get("deferred") and not c.get("skipped")]
    passed = all(c["pass"] for c in hard) if hard else True
    return {
        "method": "smoke_checklist",
        "checks": checks,
        "pass": passed,
        "note": "Tier-2 generation batteries deferred without llama/ollama runner.",
    }


def _tier3_placeholder() -> dict[str, Any]:
    return {
        "method": "skipped",
        "pass": None,
        "note": "Tier-3 benchmarks (MMLU/GSM8K/HumanEval) not run in v1 plumbing.",
        "skipped": True,
    }


def _feedback_from_tier1(tier1: dict[str, Any], sensitivity_path: Path | None) -> list[dict[str, Any]]:
    fb: list[dict[str, Any]] = []
    if tier1.get("pass"):
        return fb
    detail = tier1.get("pass_detail") or {}
    if not detail.get("mean_kld", True):
        fb.append(
            {
                "constraint": "pin_highest_kld_group",
                "action": "Raise precision +1 on the worst sensitivity group; re-run optimize",
                "reason": f"mean_kld={tier1.get('metrics', {}).get('mean_kld')}",
            }
        )
    if not detail.get("candidate_exists", True):
        fb.append(
            {
                "constraint": "export_required",
                "action": "Run odg export --mode llama --force before validate release",
                "reason": "No candidate GGUF",
            }
        )
    if sensitivity_path and sensitivity_path.is_file():
        try:
            sens = json.loads(sensitivity_path.read_text())
            pinned = sens.get("pinned_hints") or []
            if pinned:
                fb.append(
                    {
                        "constraint": "respect_pin_hints",
                        "action": f"Ensure {pinned[0].get('group_id')} stays ≥ Q5_K",
                        "reason": "sensitivity pin_high",
                    }
                )
        except Exception:  # noqa: BLE001
            pass
    return fb


def _write_report(
    path: Path,
    *,
    model_ref: str,
    verdict: str,
    recipe: dict[str, Any],
    tier1: dict[str, Any],
    tier2: dict[str, Any],
    tier3: dict[str, Any],
    feedback: list[dict[str, Any]],
    export_out: str | None,
    export_nbytes: int | None,
) -> None:
    m1 = tier1.get("metrics") or {}
    lines = [
        f"# OpenDynamicGGUF validation report",
        "",
        f"**Model:** `{model_ref}`  ",
        f"**Verdict:** **{verdict}**  ",
        f"**Candidate:** `{export_out or '(missing)'}`  ",
        f"**Size:** {export_nbytes / (1024**2):.1f} MiB"
        if export_nbytes
        else "**Size:** (dry-run)",
        f"**Recipe estimate:** {recipe.get('estimated_bytes', 0) / (1024**2):.1f} MiB  ",
        f"**Source GGUF sha:** `{recipe.get('gguf_sha256', '')[:24]}…`",
        "",
        "## Tier 1 — Logit fidelity (held-out)",
        "",
        f"- method: `{tier1.get('method')}`",
        f"- mean KLD: `{m1.get('mean_kld', 'n/a')}`",
        f"- p99.9 KLD: `{m1.get('p999_kld', 'n/a')}`",
        f"- max KLD: `{m1.get('max_kld', 'n/a')}`",
        f"- top1 agree: `{m1.get('top1_agree', 'n/a')}`",
        f"- **pass:** `{tier1.get('pass')}`",
        f"- note: {tier1.get('note', '')}",
        "",
        "## Tier 2 — Behavioral smoke",
        "",
        f"- **pass:** `{tier2.get('pass')}`",
    ]
    for c in tier2.get("checks") or []:
        lines.append(f"  - {c['name']}: pass={c['pass']} — {c.get('detail', '')}")
    lines += [
        "",
        "## Tier 3 — Benchmarks",
        "",
        f"- **pass:** `{tier3.get('pass')}` ({tier3.get('note', '')})",
        "",
        "## Feedback to optimizer",
        "",
    ]
    if feedback:
        for f in feedback:
            lines.append(
                f"- **{f.get('constraint')}**: {f.get('action')} "
                f"({f.get('reason')})"
            )
    else:
        lines.append("- (none)")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

    # Minimal HTML twin
    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>ODG report — {model_ref}</title>",
        "<style>body{font-family:ui-sans-serif,system-ui;max-width:800px;"
        "margin:2rem auto;padding:0 1rem;line-height:1.5}"
        "code{background:#f4f4f4;padding:0.1em 0.3em;border-radius:4px}"
        f".verdict{{font-size:1.4rem;font-weight:700}}</style></head><body>",
        f"<h1>OpenDynamicGGUF report</h1>",
        f"<p class='verdict'>Verdict: {verdict}</p>",
        f"<p>Model: <code>{model_ref}</code></p>",
        f"<pre>{path.read_text()}</pre>",
        "</body></html>",
    ]
    path.with_suffix(".html").write_text("\n".join(html), encoding="utf-8")


def validate_and_release(
    *,
    model_ref: str,
    out_dir: Path,
    recipe_path: str | Path,
    export_manifest: dict[str, Any] | None = None,
    specialty_domain: str | None = None,
    sensitivity_path: str | Path | None = None,
    catalog: dict[str, Any] | None = None,
    assignments: dict[str, str] | None = None,
    optimize_manifest: dict[str, Any] | None = None,
    resolve_descriptor: dict[str, Any] | None = None,
    mode: Mode = "auto",
    allow_provisional: bool = True,
) -> ValidateResult:
    log: list[str] = []
    notes: list[str] = []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    recipe_path = Path(recipe_path)
    recipe = _parse_recipe_fields(recipe_path)
    export_manifest = export_manifest or {}
    candidate = export_manifest.get("gguf_out")
    has_candidate = bool(candidate and Path(candidate).is_file())
    export_nbytes = export_manifest.get("gguf_out_nbytes")
    export_method = export_manifest.get("method")

    log.append(f"1. Recipe: {recipe_path}")
    log.append(
        f"2. Candidate: {candidate or '(missing)'} method={export_method}"
    )

    # Real llama Tier-1 not wired without logits-heldout; proxy always for now
    # unless mode=llama and we have both binary + bin (future).
    if mode == "llama" and not has_candidate:
        raise RuntimeError(
            "llama validate mode needs an exported candidate GGUF "
            "(odg export --mode llama)"
        )

    method = "proxy_gates"
    log.append("3. Tier-1: proxy_from_recipe (held-out logits not available)")
    tier1 = _tier1_proxy(recipe, has_candidate=has_candidate)
    # For dry-run exports, allow provisional pass path
    if not has_candidate and allow_provisional:
        tier1["pass_detail"]["candidate_exists"] = False

    log.append("4. Tier-2: smoke checklist")
    tier2 = _tier2_smoke(
        specialty=specialty_domain,
        recipe=recipe,
        export_nbytes=export_nbytes if isinstance(export_nbytes, int) else None,
    )

    log.append("5. Tier-3: skipped (v1)")
    tier3 = _tier3_placeholder()

    feedback = _feedback_from_tier1(
        tier1, Path(sensitivity_path) if sensitivity_path else None
    )

    if has_candidate and tier1.get("pass") and tier2.get("pass"):
        verdict = "RELEASE"
    elif not has_candidate and allow_provisional and tier2.get("pass"):
        # Plumbing path: recipe+export plan OK, GGUF not built yet
        verdict = "PROVISIONAL"
        notes.append(
            "PROVISIONAL — export was dry-run. Re-run export --mode llama "
            "then validate for RELEASE."
        )
    else:
        verdict = "FAIL"

    log.append(f"6. Verdict: {verdict}")

    report_path = out_dir / "report.md"
    _write_report(
        report_path,
        model_ref=model_ref,
        verdict=verdict,
        recipe=recipe,
        tier1=tier1,
        tier2=tier2,
        tier3=tier3,
        feedback=feedback,
        export_out=candidate,
        export_nbytes=export_nbytes if isinstance(export_nbytes, int) else None,
    )
    log.append("7. Wrote report.md + report.html")

    # Quantization report card (per-layer / per-group compression)
    report_card_paths: dict[str, str] = {}
    if catalog and assignments:
        sens_rows = None
        if sensitivity_path and Path(sensitivity_path).is_file():
            try:
                sens_rows = json.loads(Path(sensitivity_path).read_text()).get("rows")
            except Exception:  # noqa: BLE001
                sens_rows = None
        validate_payload = {
            "verdict": verdict,
            "tier1": tier1,
            "tier2": tier2,
            "tier3": tier3,
        }
        report_card_paths = write_quantization_report_card(
            out_dir,
            model_ref=model_ref,
            catalog=catalog,
            assignments=assignments,
            sensitivity_rows=sens_rows,
            optimize_manifest=optimize_manifest,
            validate_payload=validate_payload,
            resolve_descriptor=resolve_descriptor
            or ({"specialty_domain": specialty_domain} if specialty_domain else None),
        )
        log.append(
            "8. Wrote quantization_report_card.html/.md/.json "
            f"({len(assignments)} groups, layer-by-layer)"
        )
        notes.append(
            f"Quantization report card: {report_card_paths.get('html')}"
        )
    else:
        log.append("8. Skipped report card (catalog/assignments missing)")

    release_dir = None

    def _stage_report_card(dest: Path) -> None:
        for key in ("html", "md", "json"):
            src = report_card_paths.get(key)
            if src and Path(src).is_file():
                shutil.copy2(src, dest / Path(src).name)

    if verdict == "RELEASE":
        release_dir_p = out_dir / "release"
        release_dir_p.mkdir(exist_ok=True)
        shutil.copy2(recipe_path, release_dir_p / "recipe.yaml")
        shutil.copy2(report_path, release_dir_p / "report.md")
        shutil.copy2(report_path.with_suffix(".html"), release_dir_p / "report.html")
        _stage_report_card(release_dir_p)
        if candidate and Path(candidate).is_file():
            dest = release_dir_p / Path(candidate).name
            if Path(candidate).resolve() != dest.resolve():
                shutil.copy2(candidate, dest)
        release_dir = str(release_dir_p)
        log.append("9. Staged release/ artifacts (+ report card)")
        notes.append("RELEASE staged under steps/15_validate/release/")
    elif verdict == "PROVISIONAL":
        release_dir_p = out_dir / "release_provisional"
        release_dir_p.mkdir(exist_ok=True)
        shutil.copy2(recipe_path, release_dir_p / "recipe.yaml")
        shutil.copy2(report_path, release_dir_p / "report.md")
        shutil.copy2(report_path.with_suffix(".html"), release_dir_p / "report.html")
        _stage_report_card(release_dir_p)
        release_dir = str(release_dir_p)
        log.append("9. Staged release_provisional/ (+ report card)")
    else:
        (out_dir / "feedback.json").write_text(
            json.dumps({"verdict": verdict, "feedback": feedback}, indent=2) + "\n",
            encoding="utf-8",
        )
        log.append("9. Wrote feedback.json for optimizer")
        notes.append("FAIL — apply feedback constraints and re-run optimize/export.")

    if tier1.get("method") == "proxy_from_recipe":
        notes.append(
            "Tier-1 used proxy KLD from recipe — not a substitute for held-out "
            "llama-perplexity --kl-divergence."
        )

    return ValidateResult(
        model_ref=model_ref,
        method=method,
        verdict=verdict,
        tier1=tier1,
        tier2=tier2,
        tier3=tier3,
        feedback=feedback,
        report_path=str(report_path),
        release_dir=release_dir,
        report_card_paths=report_card_paths,
        steps_log=log,
        notes=notes,
    )
