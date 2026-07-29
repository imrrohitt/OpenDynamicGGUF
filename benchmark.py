"""Benchmark runner — feature 02 of the platform (docs/platform/02-…).

One command turns a GGUF into comparable numbers: throughput via
``llama-bench`` (real, when the binary exists) and quality via
lm-eval-harness (real, when the package is installed — recorded as skipped
otherwise, never faked).

Rules carried over from the validation gates:
- quality deltas are **paired per-question vs a BF16 reference with a
  bootstrap confidence interval**, never raw-score comparisons;
- every result records the harness name/version and the exact task configs,
  so numbers stay comparable across machines and time.

Output is one ``benchresult.json`` per (gguf, suite) — the only evidence
format the report, model cards, and the future leaderboard accept.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from llama_bins import find_llama_binary

SCHEMA = "odg/benchresult/v1"


# ---------------------------------------------------------------------------
# Suites — data, not code (plugin evals extend this later via odg.evals)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Suite:
    id: str
    description: str
    tasks: tuple[str, ...]
    num_fewshot: int
    limit: int | None  # per-task sample cap (None = full task)


SUITES: dict[str, Suite] = {
    s.id: s
    for s in (
        Suite(
            id="smoke",
            description="Minutes-scale sanity check: tiny MMLU slice.",
            tasks=("mmlu_abstract_algebra", "mmlu_formal_logic"),
            num_fewshot=5,
            limit=32,
        ),
        Suite(
            id="standard",
            description="The comparable core: MMLU, GSM8K, HumanEval, TruthfulQA.",
            tasks=("mmlu", "gsm8k", "humaneval", "truthfulqa_mc2"),
            num_fewshot=5,
            limit=None,
        ),
        Suite(
            id="coding",
            description="Code-generation focus.",
            tasks=("humaneval", "mbpp"),
            num_fewshot=0,
            limit=None,
        ),
    )
}


# ---------------------------------------------------------------------------
# Paired statistics (shared rule: paired delta + CI, never raw thresholds)
# ---------------------------------------------------------------------------


def paired_bootstrap_delta(
    candidate: list[float] | np.ndarray,
    reference: list[float] | np.ndarray,
    *,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """
    Per-question paired delta (candidate − reference) with a bootstrap CI.

    Both arrays must be scored on the *same questions in the same order*.
    """
    cand = np.asarray(candidate, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if cand.shape != ref.shape or cand.ndim != 1 or cand.size == 0:
        raise ValueError(
            f"Paired scores must be equal-length non-empty 1-D arrays, "
            f"got {cand.shape} vs {ref.shape}"
        )
    diffs = cand - ref
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_boot, diffs.size))
    boot_means = diffs[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boot_means, [alpha, 1.0 - alpha])
    return {
        "paired_delta": float(diffs.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ci": ci,
        "n": int(diffs.size),
    }


# ---------------------------------------------------------------------------
# Throughput via llama-bench
# ---------------------------------------------------------------------------


def run_llama_bench(
    gguf_path: Path,
    *,
    llama_bench: Path | str | None = None,
    n_prompt: int = 512,
    n_gen: int = 128,
    timeout_s: int = 1800,
) -> tuple[dict[str, Any] | None, list[str]]:
    """
    Measure prompt-processing / token-generation throughput.

    Returns (throughput dict | None, log lines). None when the binary is
    missing or the run fails — the caller records the skip honestly.
    """
    log: list[str] = []
    binary = find_llama_binary("llama-bench", llama_bench)
    if binary is None:
        log.append("llama-bench not found (set LLAMA_CPP_DIR) — throughput skipped")
        return None, log

    cmd = [
        str(binary),
        "-m",
        str(gguf_path),
        "-p",
        str(n_prompt),
        "-n",
        str(n_gen),
        "-o",
        "json",
    ]
    log.append(f"Running: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        log.append(f"llama-bench timed out after {timeout_s}s — throughput skipped")
        return None, log
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        log.append(f"llama-bench failed (exit {proc.returncode}): {' / '.join(tail)}")
        return None, log

    result = parse_llama_bench_json(proc.stdout)
    if result is None:
        log.append("Could not parse llama-bench JSON output — throughput skipped")
        return None, log
    log.append(
        f"Throughput: pp {result.get('pp_tps', '-')} t/s · tg {result.get('tg_tps', '-')} t/s"
    )
    return result, log


def parse_llama_bench_json(stdout: str) -> dict[str, Any] | None:
    """Extract pp/tg tokens-per-second from ``llama-bench -o json`` output."""
    start = stdout.find("[")
    end = stdout.rfind("]")
    if start < 0 or end <= start:
        return None
    try:
        entries = json.loads(stdout[start : end + 1])
    except json.JSONDecodeError:
        return None
    out: dict[str, Any] = {"measured": True, "tool": "llama-bench"}
    for e in entries:
        if not isinstance(e, dict):
            continue
        tps = e.get("avg_ts")
        if tps is None:
            continue
        if int(e.get("n_prompt") or 0) > 0 and int(e.get("n_gen") or 0) == 0:
            out["pp_tps"] = round(float(tps), 2)
        elif int(e.get("n_gen") or 0) > 0:
            out["tg_tps"] = round(float(tps), 2)
        if "backend" not in out and e.get("backends"):
            out["backend"] = e["backends"]
        if "model_type" not in out and e.get("model_type"):
            out["model_type"] = e["model_type"]
    return out if ("pp_tps" in out or "tg_tps" in out) else None


# ---------------------------------------------------------------------------
# Quality via lm-eval-harness (optional dependency; honest skip otherwise)
# ---------------------------------------------------------------------------


def run_quality(
    gguf_path: Path,
    suite: Suite,
    *,
    reference_scores: dict[str, list[float]] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """
    Run the suite's tasks through lm-eval-harness against the GGUF.

    Requires the optional ``lm-eval`` package with a llama.cpp-capable model
    backend. When unavailable, returns an explicitly skipped record — quality
    numbers are never estimated.

    ``reference_scores``: per-task per-question scores of the BF16 reference
    (from a previous run on the same tasks) — enables paired deltas.
    """
    log: list[str] = []
    try:
        import lm_eval  # type: ignore[import-not-found]
    except ImportError:
        log.append(
            "lm-eval not installed — quality tasks skipped "
            "(pip install lm-eval[api] to enable)"
        )
        return {
            "skipped": True,
            "reason": "lm-eval-harness not installed",
            "tasks": {},
        }, log

    harness_version = getattr(lm_eval, "__version__", "unknown")
    log.append(f"lm-eval {harness_version}: tasks={list(suite.tasks)}")
    try:
        results = lm_eval.simple_evaluate(  # type: ignore[attr-defined]
            model="gguf",
            model_args=f"base_url=None,model={gguf_path}",
            tasks=list(suite.tasks),
            num_fewshot=suite.num_fewshot,
            limit=suite.limit,
            log_samples=True,
        )
    except Exception as exc:  # noqa: BLE001 — third-party surface is wide
        log.append(f"lm-eval run failed: {exc}")
        return {
            "skipped": True,
            "reason": f"lm-eval run failed: {exc}",
            "tasks": {},
        }, log

    tasks_out: dict[str, Any] = {}
    for task_id, metrics in (results.get("results") or {}).items():
        score = None
        for key in ("acc,none", "acc", "exact_match,none", "pass@1", "mc2"):
            if key in metrics:
                score = float(metrics[key])
                break
        entry: dict[str, Any] = {"score": score, "metrics": metrics}
        per_q = _per_question_scores(results, task_id)
        if per_q and reference_scores and task_id in reference_scores:
            ref = reference_scores[task_id]
            if len(ref) == len(per_q):
                entry.update(paired_bootstrap_delta(per_q, ref))
        tasks_out[task_id] = entry
        log.append(f"  {task_id}: score={score}")

    return {
        "skipped": False,
        "harness": {"name": "lm-eval", "version": harness_version},
        "num_fewshot": suite.num_fewshot,
        "limit": suite.limit,
        "tasks": tasks_out,
    }, log


def _per_question_scores(results: dict[str, Any], task_id: str) -> list[float] | None:
    samples = (results.get("samples") or {}).get(task_id)
    if not samples:
        return None
    scores: list[float] = []
    for s in samples:
        for key in ("acc", "exact_match", "pass"):
            if key in s:
                scores.append(float(s[key]))
                break
    return scores or None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    model_ref: str | None
    gguf_path: str
    gguf_sha256: str
    suite: str
    quality: dict[str, Any]
    throughput: dict[str, Any] | None
    memory: dict[str, Any]
    device_profile: dict[str, Any] | None
    recipe_sha256: str | None
    created_at: str
    result_path: str | None = None
    steps_log: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["schema"] = SCHEMA
        return d


def sha256_file(path: Path, chunk_mb: int = 8) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_mb * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def run_benchmark(
    gguf_path: Path,
    *,
    suite_id: str = "smoke",
    model_ref: str | None = None,
    device_profile: dict[str, Any] | None = None,
    recipe_path: Path | None = None,
    reference_scores: dict[str, list[float]] | None = None,
    llama_bench: Path | str | None = None,
    out_dir: Path | None = None,
) -> BenchmarkResult:
    """Run one suite against one GGUF and write ``benchresult.json``."""
    gguf_path = Path(gguf_path)
    if not gguf_path.is_file():
        raise FileNotFoundError(f"GGUF not found: {gguf_path}")
    if suite_id not in SUITES:
        known = ", ".join(sorted(SUITES))
        raise ValueError(f"Unknown suite {suite_id!r}. Suites: {known}")
    suite = SUITES[suite_id]

    log: list[str] = [f"1. Suite {suite.id!r}: {suite.description}"]
    notes: list[str] = []

    gguf_sha = sha256_file(gguf_path)
    log.append(f"2. GGUF sha256={gguf_sha[:16]}… ({gguf_path.name})")

    recipe_sha = None
    if recipe_path and Path(recipe_path).is_file():
        recipe_sha = sha256_file(Path(recipe_path))
        log.append(f"3. Recipe sha256={recipe_sha[:16]}…")

    throughput, tp_log = run_llama_bench(gguf_path, llama_bench=llama_bench)
    log.extend(f"4. {line}" for line in tp_log)
    if throughput is None:
        notes.append("Throughput not measured — llama-bench unavailable or failed.")
    elif device_profile:
        throughput["device"] = device_profile.get("id")

    quality, q_log = run_quality(gguf_path, suite, reference_scores=reference_scores)
    log.extend(f"5. {line}" for line in q_log)
    if quality.get("skipped"):
        notes.append(f"Quality tasks skipped: {quality.get('reason')}")
    elif not reference_scores:
        notes.append(
            "No BF16 reference scores supplied — raw scores only, no paired "
            "deltas. Benchmark the reference GGUF first and pass --reference-scores."
        )

    result = BenchmarkResult(
        model_ref=model_ref,
        gguf_path=str(gguf_path),
        gguf_sha256=gguf_sha,
        suite=suite.id,
        quality=quality,
        throughput=throughput,
        memory={
            "weights_bytes": gguf_path.stat().st_size,
            "weights_gb": round(gguf_path.stat().st_size / (1024**3), 3),
        },
        device_profile=device_profile,
        recipe_sha256=recipe_sha,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        steps_log=log,
        notes=notes,
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "benchresult.json"
        out_path.write_text(json.dumps(result.summary_dict(), indent=2) + "\n")
        result.result_path = str(out_path)
        result.steps_log.append(f"6. Wrote {out_path}")

    return result


def find_run_benchresults(run_root: Path) -> list[dict[str, Any]]:
    """All benchresult.json files stored under a run (newest first)."""
    bench_dir = Path(run_root) / "benchmarks"
    if not bench_dir.is_dir():
        return []
    out = []
    for p in sorted(bench_dir.glob("*/benchresult.json"), reverse=True):
        try:
            out.append(json.loads(p.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return out
