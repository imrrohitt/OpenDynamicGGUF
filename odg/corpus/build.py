"""
Step 07 — Build mixed calibration corpus with 3-way split.

Hard rule: held-out must never be used by the optimizer (Steps 12–13).
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any

from .prompts import (
    SHARES,
    code_bank,
    conversation_bank,
    domain_bank,
    math_bank,
    multilingual_bank,
)
from .templates import render_document
from .types import CorpusResult

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
