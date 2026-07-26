"""CLI for OpenDynamicGGUF."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="odg",
        description="OpenDynamicGGUF — automatic dynamic GGUF quantization",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=None,
        help="Artifacts root for checkpoints (default: ./artifacts)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- resolve (step 01) ---
    p_resolve = sub.add_parser(
        "resolve",
        help="Step 01: resolve model ref + checkpoint to filesystem store",
    )
    p_resolve.add_argument("--model", "-m", required=True)
    p_resolve.add_argument("--download-weights", action="store_true")
    p_resolve.add_argument(
        "--prefer-hf",
        action="store_true",
        help="For Ollama tags: use Hugging Face BF16 instead of local Ollama GGUF",
    )
    p_resolve.add_argument(
        "--run",
        default=None,
        help="Existing run_id to resume into (default: model's CURRENT run or new)",
    )
    p_resolve.add_argument(
        "--new-run",
        action="store_true",
        help="Always create a fresh run (do not resume CURRENT)",
    )
    p_resolve.add_argument(
        "--force",
        action="store_true",
        help="Re-run resolve even if already checkpointed as done",
    )
    p_resolve.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional extra copy of output JSON",
    )
    p_resolve.add_argument("--cache-dir", type=Path, default=None)
    p_resolve.add_argument("--no-explain", action="store_true")

    # --- load (step 02) ---
    p_load = sub.add_parser(
        "load",
        help="Step 02: open resolved model (GGUF/HF) and checkpoint tensor index",
    )
    p_load.add_argument("--model", "-m", default=None, help="Model ref (uses CURRENT run)")
    p_load.add_argument("--run", default=None, help="run_id (default: CURRENT for --model)")
    p_load.add_argument("--force", action="store_true", help="Re-run even if done")
    p_load.add_argument("--no-explain", action="store_true")

    # --- enumerate (step 03) ---
    p_enum = sub.add_parser(
        "enumerate",
        help="Step 03: list every tensor (name/shape/dtype/nbytes) from Step 02 index",
    )
    p_enum.add_argument("--model", "-m", default=None)
    p_enum.add_argument("--run", default=None)
    p_enum.add_argument("--force", action="store_true")
    p_enum.add_argument("--no-explain", action="store_true")

    # --- classify (step 04) ---
    p_cls = sub.add_parser(
        "classify",
        help="Step 04: assign role / depth / quantizable to every tensor",
    )
    p_cls.add_argument("--model", "-m", default=None)
    p_cls.add_argument("--run", default=None)
    p_cls.add_argument("--force", action="store_true")
    p_cls.add_argument("--no-explain", action="store_true")

    # --- status / runs ---
    p_status = sub.add_parser("status", help="Show checkpoint status for a run")
    p_status.add_argument("--run", default=None, help="run_id (default: latest for --model)")
    p_status.add_argument("--model", "-m", default=None)

    p_runs = sub.add_parser("runs", help="List all checkpointed runs")

    args = parser.parse_args(argv)

    if args.command == "resolve":
        return cmd_resolve(args)
    if args.command == "load":
        return cmd_load(args)
    if args.command == "enumerate":
        return cmd_enumerate(args)
    if args.command == "classify":
        return cmd_classify(args)
    if args.command == "status":
        return cmd_status(args)
    if args.command == "runs":
        return cmd_runs(args)

    parser.error(f"Unknown command {args.command}")
    return 2


def _store(args: argparse.Namespace):
    from odg.store import RunStore

    return RunStore(args.artifacts)


def cmd_resolve(args: argparse.Namespace) -> int:
    from odg.resolve import resolve_model
    from odg.store import StepAlreadyDone

    print_explain = not args.no_explain
    store = _store(args)

    meta = store.get_or_create_run(
        args.model,
        run_id=args.run,
        resume=not args.new_run,
    )

    if print_explain:
        _banner(args.model, meta.run_id, meta.root)

    # Skip if already done
    if store.is_step_done(meta.run_id, "resolve") and not args.force:
        out = store.read_step_output(meta.run_id, "resolve")
        if print_explain:
            print("Step 01 already checkpointed as done — loading from store.")
            print(f"  run   : {meta.run_id}")
            print(f"  path  : {store.step_path(meta.run_id, 'resolve')}")
            print("  (pass --force to re-run)")
            print("\n=== stored output ===")
            print(json.dumps(out, indent=2))
            print("\n" + store.summary(meta.run_id))
        elif out:
            print(json.dumps(out, indent=2))
        return 0

    input_data = {
        "model": args.model,
        "prefer_hf": args.prefer_hf,
        "download_weights": args.download_weights,
    }

    try:
        step_dir = store.begin_step(
            meta.run_id, "resolve", input_data, force=args.force
        )
    except StepAlreadyDone:
        # race / edge — treat as skip
        return cmd_resolve(args)

    try:
        result = resolve_model(
            args.model,
            cache_dir=args.cache_dir,
            download_weights=args.download_weights,
            prefer_hf=args.prefer_hf,
        )
    except Exception as exc:  # noqa: BLE001
        store.fail_step(meta.run_id, "resolve", str(exc))
        print(f"\nERROR in Step 01 resolve: {exc}", file=sys.stderr)
        print(f"Checkpointed failure → {store.step_path(meta.run_id, 'resolve')}")
        return 1

    payload = result.to_dict()
    log_text = "\n".join(result.steps_log) + "\n"
    store.complete_step(
        meta.run_id,
        "resolve",
        payload,
        log_text=log_text,
    )

    if print_explain:
        _explain(result)
        print(f"\n=== checkpoint saved ===")
        print(f"  run_id   : {meta.run_id}")
        print(f"  step dir : {step_dir}")
        print(f"  files    : input.json, output.json, status.json, log.txt")
        print("\n" + store.summary(meta.run_id))

    text = json.dumps(payload, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        if print_explain:
            print(f"\nAlso wrote copy → {args.out}")

    if args.no_explain:
        print(text)
    else:
        print("\n=== resolve result (JSON) ===")
        print(text)

    if not result.hf_repo_id and not result.local_path:
        return 2
    return 0


def cmd_load(args: argparse.Namespace) -> int:
    from odg.load import load_model
    from odg.load.load import tensor_index_from_resolve
    from odg.store import StepAlreadyDone

    print_explain = not args.no_explain
    store = _store(args)

    try:
        meta = _require_run(store, model=args.model, run_id=args.run)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not store.is_step_done(meta.run_id, "resolve"):
        print(
            "ERROR: Step 01 (resolve) is not done for this run.\n"
            f"  Run: odg resolve --model {meta.model_ref}",
            file=sys.stderr,
        )
        return 1

    resolve_out = store.read_step_output(meta.run_id, "resolve")
    if not resolve_out:
        print("ERROR: missing resolve output.json", file=sys.stderr)
        return 1

    if print_explain:
        _banner_load(meta.model_ref, meta.run_id, meta.root)

    if store.is_step_done(meta.run_id, "load") and not args.force:
        out = store.read_step_output(meta.run_id, "load")
        if print_explain:
            print("Step 02 already checkpointed as done — loading from store.")
            print(f"  path: {store.step_path(meta.run_id, 'load')}")
            print("  (pass --force to re-run)\n")
            print(json.dumps(out, indent=2))
            print("\n" + store.summary(meta.run_id))
        elif out:
            print(json.dumps(out, indent=2))
        return 0

    input_data = {
        "from_step": "resolve",
        "local_path": resolve_out.get("local_path"),
        "source_is_quantized": resolve_out.get("source_is_quantized"),
    }

    try:
        step_dir = store.begin_step(meta.run_id, "load", input_data, force=args.force)
    except StepAlreadyDone:
        return cmd_load(args)

    try:
        loaded = load_model(resolve_out)
        # Full tensor index for Step 03 (separate artifact — can be large)
        tensors = tensor_index_from_resolve(resolve_out)
    except Exception as exc:  # noqa: BLE001
        store.fail_step(meta.run_id, "load", str(exc))
        print(f"\nERROR in Step 02 load: {exc}", file=sys.stderr)
        print(f"Checkpointed failure → {store.step_path(meta.run_id, 'load')}")
        return 1

    payload = loaded.to_dict()
    log_text = "\n".join(loaded.steps_log) + "\n"
    store.complete_step(
        meta.run_id,
        "load",
        payload,
        log_text=log_text,
        extra_artifacts={
            "tensor_index.json": json.dumps(tensors, indent=2).encode("utf-8") + b"\n",
        },
    )

    if print_explain:
        _explain_load(loaded)
        print("\n=== checkpoint saved ===")
        print(f"  run_id   : {meta.run_id}")
        print(f"  step dir : {step_dir}")
        print("  files    : input.json, output.json, status.json, log.txt, tensor_index.json")
        print("\n" + store.summary(meta.run_id))
        print("\n=== load result (JSON) ===")
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, indent=2))

    return 0


def cmd_enumerate(args: argparse.Namespace) -> int:
    from odg.enumerate import enumerate_tensors
    from odg.enumerate.enumerate import to_tsv
    from odg.store import StepAlreadyDone

    print_explain = not args.no_explain
    store = _store(args)

    try:
        meta = _require_run(store, model=args.model, run_id=args.run)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not store.is_step_done(meta.run_id, "load"):
        print(
            "ERROR: Step 02 (load) is not done.\n"
            f"  Run: odg load --model {meta.model_ref}",
            file=sys.stderr,
        )
        return 1

    if print_explain:
        _banner_enumerate(meta.model_ref, meta.run_id, meta.root)

    if store.is_step_done(meta.run_id, "enumerate") and not args.force:
        out = store.read_step_output(meta.run_id, "enumerate")
        if print_explain:
            print("Step 03 already checkpointed as done — loading from store.")
            print(f"  path: {store.step_path(meta.run_id, 'enumerate')}")
            print("  (pass --force to re-run)\n")
            print(json.dumps(out, indent=2))
            print("\n" + store.summary(meta.run_id))
        elif out:
            print(json.dumps(out, indent=2))
        return 0

    index_path = store.step_path(meta.run_id, "load") / "tensor_index.json"
    if not index_path.is_file():
        print(f"ERROR: missing {index_path}", file=sys.stderr)
        return 1
    tensor_index = json.loads(index_path.read_text())

    input_data = {
        "from_step": "load",
        "tensor_index_path": str(index_path),
        "n_tensors_in": len(tensor_index),
    }

    try:
        step_dir = store.begin_step(
            meta.run_id, "enumerate", input_data, force=args.force
        )
    except StepAlreadyDone:
        return cmd_enumerate(args)

    try:
        result = enumerate_tensors(tensor_index)
    except Exception as exc:  # noqa: BLE001
        store.fail_step(meta.run_id, "enumerate", str(exc))
        print(f"\nERROR in Step 03 enumerate: {exc}", file=sys.stderr)
        return 1

    payload = result.summary_dict()
    full = result.to_dict()
    tsv = to_tsv(result.tensors)
    log_text = "\n".join(result.steps_log) + "\n"

    store.complete_step(
        meta.run_id,
        "enumerate",
        payload,
        log_text=log_text,
        extra_artifacts={
            "tensors.json": json.dumps(full, indent=2).encode("utf-8") + b"\n",
            "tensors.tsv": tsv.encode("utf-8"),
        },
    )

    if print_explain:
        _explain_enumerate(result)
        print("\n=== checkpoint saved ===")
        print(f"  run_id   : {meta.run_id}")
        print(f"  step dir : {step_dir}")
        print("  files    : output.json, tensors.json, tensors.tsv, status.json, log.txt")
        print("\n" + store.summary(meta.run_id))
        print("\n=== enumerate summary (JSON) ===")
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, indent=2))

    return 0


def cmd_classify(args: argparse.Namespace) -> int:
    from odg.classify import classify_tensors
    from odg.store import StepAlreadyDone

    print_explain = not args.no_explain
    store = _store(args)

    try:
        meta = _require_run(store, model=args.model, run_id=args.run)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not store.is_step_done(meta.run_id, "enumerate"):
        print(
            "ERROR: Step 03 (enumerate) is not done.\n"
            f"  Run: odg enumerate --model {meta.model_ref}",
            file=sys.stderr,
        )
        return 1

    if print_explain:
        _banner_classify(meta.model_ref, meta.run_id, meta.root)

    if store.is_step_done(meta.run_id, "classify") and not args.force:
        out = store.read_step_output(meta.run_id, "classify")
        if print_explain:
            print("Step 04 already checkpointed as done — loading from store.")
            print(f"  path: {store.step_path(meta.run_id, 'classify')}")
            print("  (pass --force to re-run)\n")
            print(json.dumps(out, indent=2))
            print("\n" + store.summary(meta.run_id))
        elif out:
            print(json.dumps(out, indent=2))
        return 0

    tensors_path = store.step_path(meta.run_id, "enumerate") / "tensors.json"
    if not tensors_path.is_file():
        print(f"ERROR: missing {tensors_path}", file=sys.stderr)
        return 1
    enum_full = json.loads(tensors_path.read_text())
    tensors = enum_full.get("tensors") or []

    load_out = store.read_step_output(meta.run_id, "load") or {}
    n_layers = load_out.get("layer_count")

    input_data = {
        "from_step": "enumerate",
        "tensors_path": str(tensors_path),
        "n_tensors_in": len(tensors),
        "n_layers": n_layers,
    }

    try:
        step_dir = store.begin_step(
            meta.run_id, "classify", input_data, force=args.force
        )
    except StepAlreadyDone:
        return cmd_classify(args)

    try:
        result = classify_tensors(tensors, n_layers=n_layers)
    except Exception as exc:  # noqa: BLE001
        store.fail_step(meta.run_id, "classify", str(exc))
        print(f"\nERROR in Step 04 classify: {exc}", file=sys.stderr)
        return 1

    payload = result.summary_dict()
    full = result.to_dict()
    log_text = "\n".join(result.steps_log) + "\n"

    # TSV for browsing
    tsv_lines = [
        "index\tname\trole\tdepth\tlayer\tgroup_id\tquantizable\tdtype\tshape\tnbytes"
    ]
    for t in result.tensors:
        shape = "x".join(str(d) for d in t.shape)
        tsv_lines.append(
            f"{t.index}\t{t.name}\t{t.role}\t{t.depth or ''}\t"
            f"{'' if t.layer is None else t.layer}\t{t.group_id}\t"
            f"{int(t.quantizable)}\t{t.dtype}\t{shape}\t{t.nbytes}"
        )
    tsv = "\n".join(tsv_lines) + "\n"

    store.complete_step(
        meta.run_id,
        "classify",
        payload,
        log_text=log_text,
        extra_artifacts={
            "classified.json": json.dumps(full, indent=2).encode("utf-8") + b"\n",
            "classified.tsv": tsv.encode("utf-8"),
        },
    )

    if print_explain:
        _explain_classify(result)
        print("\n=== checkpoint saved ===")
        print(f"  run_id   : {meta.run_id}")
        print(f"  step dir : {step_dir}")
        print(
            "  files    : output.json, classified.json, classified.tsv, status.json, log.txt"
        )
        print("\n" + store.summary(meta.run_id))
        print("\n=== classify summary (JSON) ===")
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, indent=2))

    return 0


def cmd_status(args: argparse.Namespace) -> int:
    store = _store(args)
    if args.run:
        meta = store.load_run(args.run)
    elif args.model:
        meta = store.latest_run_for_model(args.model)
        if meta is None:
            print(f"No run found for model {args.model!r}", file=sys.stderr)
            return 1
    else:
        runs = store.list_runs()
        if not runs:
            print("No runs yet. Run: odg resolve --model …")
            return 0
        meta = runs[0]
    print(store.summary(meta.run_id))
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    store = _store(args)
    runs = store.list_runs()
    if not runs:
        print("No runs under", store.runs_dir)
        return 0
    print(f"{'RUN_ID':<42} {'STATUS':<10} MODEL")
    print("-" * 80)
    for m in runs:
        print(f"{m.run_id:<42} {m.status:<10} {m.model_ref}")
    return 0


def _require_run(store, *, model: str | None, run_id: str | None):
    if run_id:
        return store.load_run(run_id)
    if model:
        meta = store.latest_run_for_model(model)
        if meta is None:
            raise ValueError(
                f"No run for model {model!r}. First run: odg resolve --model {model}"
            )
        return meta
    runs = store.list_runs()
    if not runs:
        raise ValueError("No runs yet. First run: odg resolve --model …")
    return runs[0]


def _banner(model: str, run_id: str, root: str) -> None:
    print(
        """
╔══════════════════════════════════════════════════════════════╗
║  OpenDynamicGGUF — Step 01: Resolve model                   ║
╚══════════════════════════════════════════════════════════════╝
""".rstrip()
    )
    print(
        f"""
What this step does
-------------------
  Input : {model}
  Mode  : Ollama tags → use LOCAL Ollama GGUF by default (no HF login).
          Pass --prefer-hf later for original BF16 from Hugging Face.

  Checkpoint store
  ----------------
  run_id : {run_id}
  root   : {root}
  Each step writes durable files so you can resume after a crash.

  This step does NOT load the neural net and does NOT quantize.
""".rstrip()
    )


def _banner_load(model: str, run_id: str, root: str) -> None:
    print(
        """
╔══════════════════════════════════════════════════════════════╗
║  OpenDynamicGGUF — Step 02: Load model                      ║
╚══════════════════════════════════════════════════════════════╝
""".rstrip()
    )
    print(
        f"""
What this step does
-------------------
  Input : Step 01 resolve output (local_path)
  Model : {model}

  For Ollama GGUF: open the file, parse header + tensor index
  (does NOT dequantize all weights into BF16 RAM).

  Checkpoint
  ----------
  run_id : {run_id}
  root   : {root}
""".rstrip()
    )


def _banner_classify(model: str, run_id: str, root: str) -> None:
    print(
        """
╔══════════════════════════════════════════════════════════════╗
║  OpenDynamicGGUF — Step 04: Classify tensors                ║
╚══════════════════════════════════════════════════════════════╝
""".rstrip()
    )
    print(
        f"""
What this step does
-------------------
  Input : Step 03 tensors.json
  Model : {model}
  Goal  : Assign role (attn_q, ffn_up, norm, …), depth bucket,
          group_id, and quantizable flag to every tensor.

  Checkpoint
  ----------
  run_id : {run_id}
  root   : {root}
""".rstrip()
    )


def _explain_classify(result) -> None:
    print("\n=== what happened ===")
    for line in result.steps_log:
        print(f"  • {line}")

    print("\n=== verdict ===")
    print(f"  ✓ Tensors            : {result.n_tensors}")
    print(f"  ✓ Layers             : {result.n_layers}")
    print(f"  ✓ Coverage           : {result.coverage:.1%} (non-other)")
    print(f"  ✓ Role summary       : {result.role_summary}")
    print(f"  ✓ Quantizable        : {result.quantizable_summary}")
    print(f"  ✓ Probe groups       : {len(result.group_summary)}")

    print("\n=== groups (role@depth) ===")
    for gid, n in list(result.group_summary.items())[:20]:
        print(f"  {gid:28} {n:4} tensors")
    if len(result.group_summary) > 20:
        print(f"  … +{len(result.group_summary) - 20} more")

    print("\n=== sample ===")
    for t in result.tensors[:12]:
        q = "Q" if t.quantizable else "—"
        print(
            f"  [{q}] {t.role:12} {t.depth or '-':7}  {t.name}"
        )

    if result.other_names:
        print("\n  unmatched:")
        for n in result.other_names[:10]:
            print(f"    - {n}")

    print("\n=== next ===")
    print("  Step 05 — build tensor_catalog.json (HF/GGUF names + groups).")


def _banner_enumerate(model: str, run_id: str, root: str) -> None:
    print(
        """
╔══════════════════════════════════════════════════════════════╗
║  OpenDynamicGGUF — Step 03: Enumerate tensors               ║
╚══════════════════════════════════════════════════════════════╝
""".rstrip()
    )
    print(
        f"""
What this step does
-------------------
  Input : Step 02 tensor_index.json
  Model : {model}
  Goal  : Flat inventory of EVERY tensor (name, shape, dtype, nbytes).
          No roles yet — that is Step 04.

  Checkpoint
  ----------
  run_id : {run_id}
  root   : {root}
""".rstrip()
    )


def _explain_enumerate(result) -> None:
    print("\n=== what happened ===")
    for line in result.steps_log:
        print(f"  • {line}")

    print("\n=== verdict ===")
    print(f"  ✓ Tensors            : {result.n_tensors}")
    print(f"  ✓ Total elements     : {result.total_elements:,}")
    print(f"  ✓ Approx nbytes      : {result.total_nbytes:,}")
    print(f"  ✓ Dtype summary      : {result.dtype_summary}")

    print("\n=== by layer (tensor counts) ===")
    # show compact: global + first/last few layers
    items = list(result.layer_summary.items())
    for k, v in items[:6]:
        print(f"  layer {k:>6}: {v} tensors")
    if len(items) > 8:
        print("  …")
        for k, v in items[-2:]:
            print(f"  layer {k:>6}: {v} tensors")

    print("\n=== sample (sorted by name) ===")
    for t in result.tensors[:10]:
        shape = "×".join(str(d) for d in t.shape)
        print(f"  {t.name:42} {shape:16} {t.dtype:6}  {t.nbytes:>10,} B")

    print("\n=== next ===")
    print("  Step 04 — classify each tensor into a role (attn_q, ffn_up, norm, …).")


def _explain_load(loaded) -> None:
    print("\n=== what happened ===")
    for line in loaded.steps_log:
        print(f"  • {line}")

    print("\n=== verdict ===")
    print(f"  ✓ Backend            : {loaded.backend}")
    print(f"  ✓ Source path        : {loaded.source_path}")
    print(f"  ✓ Tensors indexed    : {loaded.n_tensors}")
    print(f"  ✓ File size          : {loaded.file_size_bytes:,} bytes")
    print(f"  ✓ Architecture       : {loaded.architecture}")
    print(f"  ✓ Layers / embed     : {loaded.layer_count} / {loaded.embedding_length}")
    print(f"  ✓ Vocab size         : {loaded.vocab_size}")
    print(f"  ✓ Quantized source?  : {loaded.source_is_quantized}")
    print(f"  ✓ Dtype summary      : {loaded.dtype_summary}")
    if loaded.sample_tensors:
        print("\n=== sample tensors ===")
        for t in loaded.sample_tensors[:8]:
            print(f"  {t.name:40} shape={t.shape}  dtype={t.dtype}")
    if loaded.notes:
        print("\n  notes:")
        for n in loaded.notes:
            print(f"    - {n}")
    print("\n=== next ===")
    print("  Step 03 — enumerate/classify every tensor into the catalog.")


def _explain(result) -> None:
    print("\n=== what happened ===")
    for line in result.steps_log:
        print(f"  • {line}")

    print("\n=== verdict ===")
    print(f"  ✓ Kind                : {result.kind.value}")
    print(f"  ✓ Working local_path  : {result.local_path}")
    print(f"  ✓ Weights ready       : {result.weights_ready}")
    print(f"  ✓ Source quantized?   : {result.source_is_quantized}")
    print(f"  ✓ Upstream HF (later) : {result.hf_repo_id}")
    if result.rejected_quantized_source:
        print(f"  · Note: {result.rejected_quantized_source}")

    d = result.descriptor
    print("\n=== architecture descriptor ===")
    print(f"  family            : {d.family}")
    print(f"  layer_count       : {d.layer_count}")
    print(f"  embedding_length  : {d.embedding_length}")
    print(f"  parameter_count   : {d.parameter_count}")
    print(f"  context_length    : {d.context_length}")
    print(f"  specialty_domain  : {d.specialty_domain}")
    print(f"  ollama_quant      : {d.ollama_quantization}")
    if d.notes:
        print("  notes:")
        for n in d.notes:
            print(f"    - {n}")

    print("\n=== next ===")
    if result.weights_ready and result.local_path:
        print("  Local weights ready → proceed to Step 02 (inspect / catalog).")
        if result.source_is_quantized:
            print(
                "  Warning: source is already quantized. Fine for pipeline plumbing;\n"
                "  for real dynamic quant quality, later switch to --prefer-hf BF16."
            )
    else:
        print(
            "  Weights not ready. For Ollama tags, re-run without --prefer-hf,\n"
            "  or login to Hugging Face and use --prefer-hf --download-weights."
        )


if __name__ == "__main__":
    raise SystemExit(main())
