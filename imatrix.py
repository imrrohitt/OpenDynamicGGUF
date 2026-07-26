"""Step 10 — importance matrix (llama-imatrix or proxy)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import json
from pathlib import Path
import os
import shutil
import subprocess
import hashlib
from typing import Any, Literal


# --- from imatrix/types.py ---
@dataclass
class ImatrixResult:
    model_ref: str
    method: str  # "llama_imatrix" | "proxy_importance"
    gguf_path: str
    gguf_sha256: str | None
    calib_path: str
    imatrix_path: str | None
    imatrix_sha256: str | None
    proxy_path: str | None
    n_chunks: int | None
    n_tensors_scored: int
    steps_log: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary_dict(self) -> dict[str, Any]:
        return asdict(self)

# --- from imatrix/proxy.py ---
def build_proxy_importance(
    catalog: dict[str, Any],
    *,
    out_path: Path,
    gguf_sha256: str | None,
    calib_path: str,
) -> dict[str, Any]:
    """
    Write imatrix_proxy.json with per-tensor importance in [0, 1].
    """
    tensors = catalog.get("tensors") or {}
    scores: dict[str, dict[str, Any]] = {}
    raw: list[tuple[str, float]] = []

    for name, t in tensors.items():
        if not t.get("quantizable", True):
            continue
        wf = t.get("weight_features") or {}
        af = t.get("activation_features") or {}
        w_out = float(wf.get("outlier_ratio") or 0.0)
        w_var = float(wf.get("variance") or 0.0)
        a_abs = float(af.get("absmax") or 0.0)
        a_out = float(af.get("outlier_ratio") or 0.0)
        # Higher = protect more when rounding
        score = (
            2.0 * w_out
            + 0.15 * (w_var ** 0.5)
            + 0.05 * a_abs
            + 1.5 * a_out
        )
        role = str(t.get("role") or "")
        if role in {"attn_q", "attn_k", "attn_v", "attn_o"}:
            score *= 1.15
        if role in {"ffn_gate", "ffn_up"}:
            score *= 1.05
        raw.append((name, score))

    if not raw:
        payload = {
            "method": "proxy_importance",
            "gguf_sha256": gguf_sha256,
            "calib_path": calib_path,
            "n_tensors": 0,
            "tensors": {},
            "note": "No quantizable tensors found in catalog",
        }
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return payload

    mx = max(s for _, s in raw) or 1.0
    for name, score in raw:
        t = tensors[name]
        scores[name] = {
            "importance": float(score / mx),
            "raw_score": float(score),
            "role": t.get("role"),
            "group_id": t.get("group_id"),
            "quantizable": True,
        }

    # Group aggregates
    groups: dict[str, list[float]] = {}
    for name, info in scores.items():
        gid = str(info.get("group_id") or "unknown")
        groups.setdefault(gid, []).append(float(info["importance"]))
    group_scores = {
        gid: {
            "importance_mean": sum(v) / len(v),
            "importance_max": max(v),
            "n_tensors": len(v),
        }
        for gid, v in groups.items()
    }

    ranked = sorted(
        ({"name": n, **info} for n, info in scores.items()),
        key=lambda r: r["importance"],
        reverse=True,
    )

    payload = {
        "method": "proxy_importance",
        "gguf_sha256": gguf_sha256,
        "calib_path": calib_path,
        "n_tensors": len(scores),
        "tensors": scores,
        "groups": group_scores,
        "top_important": ranked[:15],
        "least_important": list(reversed(ranked[-10:])),
        "note": (
            "Proxy only — not usable as llama-quantize --imatrix. "
            "Install llama-imatrix and re-run with --mode llama."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload

# --- from imatrix/runner.py ---
def find_llama_imatrix(explicit: str | Path | None = None) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.is_file() and os.access(p, os.X_OK) else (
            p if p.is_file() else None
        )
    which = shutil.which("llama-imatrix")
    if which:
        return Path(which)
    env = os.environ.get("LLAMA_CPP_DIR") or os.environ.get("LLAMA_CPP")
    candidates: list[Path] = []
    if env:
        root = Path(env).expanduser()
        candidates += [
            root / "llama-imatrix",
            root / "build" / "bin" / "llama-imatrix",
            root / "bin" / "llama-imatrix",
        ]
    home = Path.home()
    candidates += [
        home / "llama.cpp" / "build" / "bin" / "llama-imatrix",
        home / "llama.cpp" / "llama-imatrix",
        Path("/opt/homebrew/bin/llama-imatrix"),
        Path("/usr/local/bin/llama-imatrix"),
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def run_llama_imatrix(
    *,
    binary: Path,
    model_gguf: Path,
    calib_txt: Path,
    outfile: Path,
    n_chunks: int | None = None,
    extra_args: list[str] | None = None,
) -> str:
    """
    Run llama-imatrix. Returns combined stdout+stderr tail for the log.
    """
    outfile.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(binary),
        "-m",
        str(model_gguf),
        "-f",
        str(calib_txt),
        "-o",
        str(outfile),
    ]
    if n_chunks is not None and n_chunks > 0:
        # Flag name varies slightly by version; --chunks is common
        cmd += ["--chunks", str(n_chunks)]
    if extra_args:
        cmd += list(extra_args)

    proc = subprocess.run(cmd, capture_output=True, text=True)
    log = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        raise RuntimeError(
            f"llama-imatrix failed (exit {proc.returncode}):\n"
            f"cmd: {' '.join(cmd)}\n"
            f"{log[-4000:]}"
        )
    if not outfile.is_file():
        raise RuntimeError(
            f"llama-imatrix exited 0 but outfile missing: {outfile}\n{log[-2000:]}"
        )
    return log

# --- from imatrix/build.py ---
Mode = Literal["auto", "llama", "proxy"]


def _sha256_file(path: Path, *, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def build_imatrix(
    *,
    model_ref: str,
    out_dir: Path,
    gguf_path: str | Path,
    calib_path: str | Path,
    catalog: dict[str, Any] | None = None,
    gguf_sha256: str | None = None,
    mode: Mode = "auto",
    llama_imatrix: str | Path | None = None,
    n_chunks: int | None = 64,
) -> ImatrixResult:
    """
    Produce imatrix.gguf (real) and/or imatrix_proxy.json under out_dir.
    """
    log: list[str] = []
    notes: list[str] = []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gguf = Path(gguf_path).expanduser().resolve()
    calib = Path(calib_path).expanduser().resolve()
    if not gguf.is_file():
        raise FileNotFoundError(f"Frozen GGUF not found: {gguf}")
    if not calib.is_file():
        raise FileNotFoundError(f"calib.txt not found: {calib}")

    if gguf_sha256 is None:
        gguf_sha256 = _sha256_file(gguf)
    log.append(f"1. GGUF: {gguf} sha256={gguf_sha256[:16]}…")
    log.append(f"2. Calib: {calib} (must NOT be heldout)")

    binary = find_llama_imatrix(llama_imatrix)
    want_llama = mode in {"auto", "llama"}

    imatrix_path: Path | None = None
    imatrix_sha: str | None = None
    proxy_path: Path | None = None
    method: str
    n_scored = 0

    if want_llama and binary is not None:
        imatrix_path = out_dir / "imatrix.gguf"
        log.append(f"3. Running llama-imatrix: {binary}")
        if n_chunks:
            log.append(f"4. chunks={n_chunks}")
        try:
            run_log = run_llama_imatrix(
                binary=binary,
                model_gguf=gguf,
                calib_txt=calib,
                outfile=imatrix_path,
                n_chunks=n_chunks,
            )
            (out_dir / "llama-imatrix.log").write_text(run_log, encoding="utf-8")
            imatrix_sha = _sha256_file(imatrix_path)
            method = "llama_imatrix"
            log.append(f"5. Wrote {imatrix_path} sha256={imatrix_sha[:16]}…")
            notes.append("Real imatrix.gguf ready for llama-quantize --imatrix.")
            # Also write proxy ranking for our optimizer readability
            if catalog:
                proxy_path = out_dir / "imatrix_proxy.json"
                payload = build_proxy_importance(
                    catalog,
                    out_path=proxy_path,
                    gguf_sha256=gguf_sha256,
                    calib_path=str(calib),
                )
                n_scored = int(payload.get("n_tensors") or 0)
                log.append(f"6. Also wrote readable proxy ranking ({n_scored} tensors)")
        except Exception:
            if mode == "llama":
                raise
            log.append("5. llama-imatrix failed — falling back to proxy_importance")
            imatrix_path = None
            method = "proxy_importance"
            if not catalog:
                raise RuntimeError(
                    "llama-imatrix failed and no catalog for proxy fallback"
                )
            proxy_path = out_dir / "imatrix_proxy.json"
            payload = build_proxy_importance(
                catalog,
                out_path=proxy_path,
                gguf_sha256=gguf_sha256,
                calib_path=str(calib),
            )
            n_scored = int(payload.get("n_tensors") or 0)
            notes.append(
                "llama-imatrix failed; proxy_importance written. "
                "Not usable as --imatrix until real binary succeeds."
            )
    elif mode == "llama":
        raise RuntimeError(
            "llama-imatrix not found. Install llama.cpp and ensure llama-imatrix "
            "is on PATH, or set LLAMA_CPP_DIR / --llama-imatrix."
        )
    else:
        method = "proxy_importance"
        if binary is None:
            log.append("3. llama-imatrix not found — using proxy_importance")
        else:
            log.append("3. mode=proxy — skipping llama-imatrix")
        if not catalog:
            raise RuntimeError("catalog required for proxy_importance mode")
        proxy_path = out_dir / "imatrix_proxy.json"
        payload = build_proxy_importance(
            catalog,
            out_path=proxy_path,
            gguf_sha256=gguf_sha256,
            calib_path=str(calib),
        )
        n_scored = int(payload.get("n_tensors") or 0)
        log.append(f"4. Wrote {proxy_path} ({n_scored} tensors scored)")
        notes.append(
            "Proxy only — install llama-imatrix and re-run with --mode llama "
            "for a real imatrix.gguf."
        )
        # Marker so later steps know what's missing
        (out_dir / "imatrix.gguf.MISSING").write_text(
            "Run: odg imatrix --mode llama --force\n", encoding="utf-8"
        )

    return ImatrixResult(
        model_ref=model_ref,
        method=method,
        gguf_path=str(gguf),
        gguf_sha256=gguf_sha256,
        calib_path=str(calib),
        imatrix_path=str(imatrix_path) if imatrix_path and imatrix_path.is_file() else None,
        imatrix_sha256=imatrix_sha,
        proxy_path=str(proxy_path) if proxy_path and proxy_path.is_file() else None,
        n_chunks=n_chunks if method == "llama_imatrix" else None,
        n_tensors_scored=n_scored,
        steps_log=log,
        notes=notes,
    )
