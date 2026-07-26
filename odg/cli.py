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

    # --- catalog (step 05) ---
    p_cat = sub.add_parser(
        "catalog",
        help="Step 05: build tensor_catalog.json (source of truth for probes)",
    )
    p_cat.add_argument("--model", "-m", default=None)
    p_cat.add_argument("--run", default=None)
    p_cat.add_argument("--force", action="store_true")
    p_cat.add_argument("--no-explain", action="store_true")

    # --- weight-features (step 06) ---
    p_wf = sub.add_parser(
        "weight-features",
        help="Step 06: compute weight stats (mean/var/sparsity/outliers/norms)",
    )
    p_wf.add_argument("--model", "-m", default=None)
    p_wf.add_argument("--run", default=None)
    p_wf.add_argument("--force", action="store_true")
    p_wf.add_argument(
        "--only-quantizable",
        action="store_true",
        help="Skip non-quantizable tensors (norms, etc.)",
    )
    p_wf.add_argument("--no-explain", action="store_true")

    # --- corpus (step 07) ---
    p_corp = sub.add_parser(
        "corpus",
        help="Step 07: build calib/search/heldout text corpus (3-way split)",
    )
    p_corp.add_argument("--model", "-m", default=None)
    p_corp.add_argument("--run", default=None)
    p_corp.add_argument("--force", action="store_true")
    p_corp.add_argument(
        "--target-tokens",
        type=int,
        default=50_000,
        help="Approximate token budget (chars/4). Default 50000; use 300000+ for production",
    )
    p_corp.add_argument("--seed", type=int, default=42)
    p_corp.add_argument("--no-explain", action="store_true")

    # --- activation-features (step 08) ---
    p_af = sub.add_parser(
        "activation-features",
        help="Step 08: activation stats from calib (forward hooks or proxy)",
    )
    p_af.add_argument("--model", "-m", default=None)
    p_af.add_argument("--run", default=None)
    p_af.add_argument("--force", action="store_true")
    p_af.add_argument(
        "--mode",
        choices=("auto", "forward", "proxy"),
        default="auto",
        help="auto: forward if BF16+torch available, else proxy (default)",
    )
    p_af.add_argument(
        "--max-docs",
        type=int,
        default=32,
        help="Max calib docs for forward pass (default 32)",
    )
    p_af.add_argument("--no-explain", action="store_true")

    # --- freeze-gguf (step 09) ---
    p_fz = sub.add_parser(
        "freeze-gguf",
        help="Step 09: freeze hashed GGUF reference (BF16 convert or promote source)",
    )
    p_fz.add_argument("--model", "-m", default=None)
    p_fz.add_argument("--run", default=None)
    p_fz.add_argument("--force", action="store_true")
    p_fz.add_argument(
        "--mode",
        choices=("auto", "hf-convert", "promote"),
        default="auto",
        help="auto: HF→BF16 if possible, else promote working GGUF",
    )
    p_fz.add_argument(
        "--convert-script",
        type=Path,
        default=None,
        help="Path to llama.cpp convert_hf_to_gguf.py (or set LLAMA_CPP_DIR)",
    )
    p_fz.add_argument(
        "--require-bf16",
        action="store_true",
        help="Fail unless the frozen file is BF16/F16 (not Q8)",
    )
    p_fz.add_argument("--no-explain", action="store_true")

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
    if args.command == "catalog":
        return cmd_catalog(args)
    if args.command == "weight-features":
        return cmd_weight_features(args)
    if args.command == "corpus":
        return cmd_corpus(args)
    if args.command == "activation-features":
        return cmd_activation_features(args)
    if args.command == "freeze-gguf":
        return cmd_freeze_gguf(args)
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


def cmd_catalog(args: argparse.Namespace) -> int:
    import hashlib

    from odg.catalog import build_catalog
    from odg.store import StepAlreadyDone

    print_explain = not args.no_explain
    store = _store(args)

    try:
        meta = _require_run(store, model=args.model, run_id=args.run)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not store.is_step_done(meta.run_id, "classify"):
        print(
            "ERROR: Step 04 (classify) is not done.\n"
            f"  Run: odg classify --model {meta.model_ref}",
            file=sys.stderr,
        )
        return 1

    if print_explain:
        _banner_catalog(meta.model_ref, meta.run_id, meta.root)

    if store.is_step_done(meta.run_id, "catalog") and not args.force:
        out = store.read_step_output(meta.run_id, "catalog")
        if print_explain:
            print("Step 05 already checkpointed as done — loading from store.")
            print(f"  path: {store.step_path(meta.run_id, 'catalog')}")
            print("  (pass --force to re-run)\n")
            print(json.dumps(out, indent=2))
            print("\n" + store.summary(meta.run_id))
        elif out:
            print(json.dumps(out, indent=2))
        return 0

    classified_path = store.step_path(meta.run_id, "classify") / "classified.json"
    if not classified_path.is_file():
        print(f"ERROR: missing {classified_path}", file=sys.stderr)
        return 1
    classified = json.loads(classified_path.read_text())
    tensors = classified.get("tensors") or []

    resolve_out = store.read_step_output(meta.run_id, "resolve") or {}
    load_out = store.read_step_output(meta.run_id, "load") or {}

    source_path = load_out.get("source_path") or resolve_out.get("local_path")
    source_sha = resolve_out.get("source_sha256")
    if source_path and not source_sha:
        try:
            h = hashlib.sha256()
            with open(source_path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            source_sha = h.hexdigest()
        except OSError:
            source_sha = None

    input_data = {
        "from_step": "classify",
        "classified_path": str(classified_path),
        "n_tensors_in": len(tensors),
        "source_path": source_path,
    }

    try:
        step_dir = store.begin_step(
            meta.run_id, "catalog", input_data, force=args.force
        )
    except StepAlreadyDone:
        return cmd_catalog(args)

    try:
        catalog = build_catalog(
            tensors,
            model_ref=meta.model_ref,
            hf_repo_id=resolve_out.get("hf_repo_id"),
            source_path=source_path,
            source_backend=load_out.get("backend"),
            source_is_quantized=bool(
                load_out.get("source_is_quantized")
                or resolve_out.get("source_is_quantized")
            ),
            source_sha256=source_sha,
            n_layers=load_out.get("layer_count") or classified.get("n_layers"),
        )
    except Exception as exc:  # noqa: BLE001
        store.fail_step(meta.run_id, "catalog", str(exc))
        print(f"\nERROR in Step 05 catalog: {exc}", file=sys.stderr)
        return 1

    payload = catalog.summary_dict()
    full = catalog.to_dict()
    log_text = "\n".join(catalog.steps_log) + "\n"

    store.complete_step(
        meta.run_id,
        "catalog",
        payload,
        log_text=log_text,
        extra_artifacts={
            "tensor_catalog.json": json.dumps(full, indent=2).encode("utf-8") + b"\n",
        },
    )

    if print_explain:
        _explain_catalog(catalog)
        print("\n=== checkpoint saved ===")
        print(f"  run_id   : {meta.run_id}")
        print(f"  step dir : {step_dir}")
        print("  files    : output.json, tensor_catalog.json, status.json, log.txt")
        print("\n" + store.summary(meta.run_id))
        print("\n=== catalog summary (JSON) ===")
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, indent=2))

    return 0


def cmd_weight_features(args: argparse.Namespace) -> int:
    from odg.store import StepAlreadyDone
    from odg.weight_features import compute_catalog_weight_features

    print_explain = not args.no_explain
    store = _store(args)

    try:
        meta = _require_run(store, model=args.model, run_id=args.run)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not store.is_step_done(meta.run_id, "catalog"):
        print(
            "ERROR: Step 05 (catalog) is not done.\n"
            f"  Run: odg catalog --model {meta.model_ref}",
            file=sys.stderr,
        )
        return 1

    if print_explain:
        _banner_weight_features(meta.model_ref, meta.run_id, meta.root)

    if store.is_step_done(meta.run_id, "weight_features") and not args.force:
        out = store.read_step_output(meta.run_id, "weight_features")
        if print_explain:
            print("Step 06 already checkpointed as done — loading from store.")
            print(f"  path: {store.step_path(meta.run_id, 'weight_features')}")
            print("  (pass --force to re-run)\n")
            print(json.dumps(out, indent=2))
            print("\n" + store.summary(meta.run_id))
        elif out:
            print(json.dumps(out, indent=2))
        return 0

    catalog_path = store.step_path(meta.run_id, "catalog") / "tensor_catalog.json"
    if not catalog_path.is_file():
        print(f"ERROR: missing {catalog_path}", file=sys.stderr)
        return 1
    catalog = json.loads(catalog_path.read_text())
    source_path = catalog.get("source_path")
    load_out = store.read_step_output(meta.run_id, "load") or {}
    if not source_path:
        source_path = load_out.get("source_path")

    input_data = {
        "from_step": "catalog",
        "catalog_path": str(catalog_path),
        "source_path": source_path,
        "only_quantizable": bool(args.only_quantizable),
        "n_tensors_in": len(catalog.get("tensors") or {}),
    }

    try:
        step_dir = store.begin_step(
            meta.run_id, "weight_features", input_data, force=args.force
        )
    except StepAlreadyDone:
        return cmd_weight_features(args)

    try:
        updated, result = compute_catalog_weight_features(
            catalog,
            source_path=source_path,
            only_quantizable=bool(args.only_quantizable),
        )
    except Exception as exc:  # noqa: BLE001
        store.fail_step(meta.run_id, "weight_features", str(exc))
        print(f"\nERROR in Step 06 weight-features: {exc}", file=sys.stderr)
        return 1

    payload = result.summary_dict()
    # Keep payload JSON-friendly / not huge: drop full group_features map from summary
    # (still written to weight_features.json + updated catalog)
    payload_out = {
        "model_ref": result.model_ref,
        "source_path": result.source_path,
        "source_is_quantized": result.source_is_quantized,
        "n_tensors": result.n_tensors,
        "n_with_features": result.n_with_features,
        "n_skipped": result.n_skipped,
        "catalog_sha256": result.catalog_sha256,
        "hardest_groups": result.hardest_groups,
        "easiest_groups": result.easiest_groups,
        "steps_log": result.steps_log,
        "notes": result.notes,
    }
    log_text = "\n".join(result.steps_log) + "\n"

    store.complete_step(
        meta.run_id,
        "weight_features",
        payload_out,
        log_text=log_text,
        extra_artifacts={
            "tensor_catalog.json": json.dumps(updated, indent=2).encode("utf-8")
            + b"\n",
            "group_features.json": json.dumps(
                result.group_features, indent=2
            ).encode("utf-8")
            + b"\n",
        },
    )

    if print_explain:
        _explain_weight_features(result)
        print("\n=== checkpoint saved ===")
        print(f"  run_id   : {meta.run_id}")
        print(f"  step dir : {step_dir}")
        print(
            "  files    : output.json, tensor_catalog.json, "
            "group_features.json, status.json, log.txt"
        )
        print("\n" + store.summary(meta.run_id))
        print("\n=== weight-features summary (JSON) ===")
        print(json.dumps(payload_out, indent=2))
    else:
        print(json.dumps(payload_out, indent=2))

    return 0


def cmd_corpus(args: argparse.Namespace) -> int:
    from odg.corpus import build_corpus
    from odg.store import StepAlreadyDone

    print_explain = not args.no_explain
    store = _store(args)

    try:
        meta = _require_run(store, model=args.model, run_id=args.run)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Corpus needs resolve descriptor; weight_features is the logical prior step
    if not store.is_step_done(meta.run_id, "weight_features"):
        print(
            "ERROR: Step 06 (weight-features) is not done.\n"
            f"  Run: odg weight-features --model {meta.model_ref}",
            file=sys.stderr,
        )
        return 1

    if print_explain:
        _banner_corpus(meta.model_ref, meta.run_id, meta.root)

    if store.is_step_done(meta.run_id, "corpus") and not args.force:
        out = store.read_step_output(meta.run_id, "corpus")
        if print_explain:
            print("Step 07 already checkpointed as done — loading from store.")
            print(f"  path: {store.step_path(meta.run_id, 'corpus')}")
            print("  (pass --force to re-run)\n")
            print(json.dumps(out, indent=2))
            print("\n" + store.summary(meta.run_id))
        elif out:
            print(json.dumps(out, indent=2))
        return 0

    resolve_out = store.read_step_output(meta.run_id, "resolve") or {}
    desc = resolve_out.get("descriptor") or {}
    chat_template = desc.get("chat_template") or "gemma3"
    specialty = desc.get("specialty_domain")

    input_data = {
        "from_step": "weight_features",
        "chat_template": chat_template,
        "specialty_domain": specialty,
        "target_tokens": int(args.target_tokens),
        "seed": int(args.seed),
        "splits": {"calib": 0.6, "search": 0.2, "heldout": 0.2},
    }

    try:
        step_dir = store.begin_step(
            meta.run_id, "corpus", input_data, force=args.force
        )
    except StepAlreadyDone:
        return cmd_corpus(args)

    try:
        result, manifest = build_corpus(
            model_ref=meta.model_ref,
            out_dir=step_dir,
            chat_template=chat_template,
            specialty_domain=specialty,
            target_tokens=int(args.target_tokens),
            seed=int(args.seed),
        )
    except Exception as exc:  # noqa: BLE001
        store.fail_step(meta.run_id, "corpus", str(exc))
        print(f"\nERROR in Step 07 corpus: {exc}", file=sys.stderr)
        return 1

    payload = result.summary_dict()
    log_text = "\n".join(result.steps_log) + "\n"

    # Files already written into step_dir by build_corpus; also store manifest
    store.complete_step(
        meta.run_id,
        "corpus",
        payload,
        log_text=log_text,
        extra_artifacts={
            "corpus_manifest.json": json.dumps(manifest, indent=2).encode("utf-8")
            + b"\n",
        },
    )

    if print_explain:
        _explain_corpus(result)
        print("\n=== checkpoint saved ===")
        print(f"  run_id   : {meta.run_id}")
        print(f"  step dir : {step_dir}")
        print(
            "  files    : output.json, calib.txt, search.txt, heldout.txt, "
            "corpus_manifest.json, status.json, log.txt"
        )
        print("\n" + store.summary(meta.run_id))
        print("\n=== corpus summary (JSON) ===")
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, indent=2))

    return 0


def cmd_activation_features(args: argparse.Namespace) -> int:
    from odg.activation_features import compute_catalog_activation_features
    from odg.store import StepAlreadyDone

    print_explain = not args.no_explain
    store = _store(args)

    try:
        meta = _require_run(store, model=args.model, run_id=args.run)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not store.is_step_done(meta.run_id, "corpus"):
        print(
            "ERROR: Step 07 (corpus) is not done.\n"
            f"  Run: odg corpus --model {meta.model_ref}",
            file=sys.stderr,
        )
        return 1
    if not store.is_step_done(meta.run_id, "weight_features"):
        print(
            "ERROR: Step 06 (weight-features) is not done.\n"
            f"  Run: odg weight-features --model {meta.model_ref}",
            file=sys.stderr,
        )
        return 1

    if print_explain:
        _banner_activation_features(meta.model_ref, meta.run_id, meta.root)

    if store.is_step_done(meta.run_id, "activation_features") and not args.force:
        out = store.read_step_output(meta.run_id, "activation_features")
        if print_explain:
            print("Step 08 already checkpointed as done — loading from store.")
            print(f"  path: {store.step_path(meta.run_id, 'activation_features')}")
            print("  (pass --force to re-run)\n")
            print(json.dumps(out, indent=2))
            print("\n" + store.summary(meta.run_id))
        elif out:
            print(json.dumps(out, indent=2))
        return 0

    catalog_path = (
        store.step_path(meta.run_id, "weight_features") / "tensor_catalog.json"
    )
    if not catalog_path.is_file():
        catalog_path = store.step_path(meta.run_id, "catalog") / "tensor_catalog.json"
    if not catalog_path.is_file():
        print(f"ERROR: missing catalog at {catalog_path}", file=sys.stderr)
        return 1

    calib_path = store.step_path(meta.run_id, "corpus") / "calib.txt"
    if not calib_path.is_file():
        print(f"ERROR: missing {calib_path}", file=sys.stderr)
        return 1

    catalog = json.loads(catalog_path.read_text())
    resolve_out = store.read_step_output(meta.run_id, "resolve") or {}
    corpus_out = store.read_step_output(meta.run_id, "corpus") or {}

    # Prefer local HF dir if resolve downloaded BF16; else hub id
    hf_local = None
    local_path = resolve_out.get("local_path")
    if local_path and Path(local_path).is_dir():
        hf_local = local_path
    hf_id = resolve_out.get("hf_repo_id")
    # Don't treat Ollama blob file as HF
    if local_path and Path(local_path).is_file():
        hf_local = None

    input_data = {
        "from_steps": ["weight_features", "corpus"],
        "catalog_path": str(catalog_path),
        "calib_path": str(calib_path),
        "mode": args.mode,
        "max_docs": int(args.max_docs),
        "hf_repo_id": hf_id,
        "hf_local_path": hf_local,
    }

    try:
        step_dir = store.begin_step(
            meta.run_id, "activation_features", input_data, force=args.force
        )
    except StepAlreadyDone:
        return cmd_activation_features(args)

    try:
        updated, result = compute_catalog_activation_features(
            catalog,
            calib_path=calib_path,
            mode=args.mode,
            hf_model_id=hf_id,
            hf_local_path=hf_local,
            max_forward_docs=int(args.max_docs),
            corpus_domain_counts=corpus_out.get("domain_counts"),
        )
    except Exception as exc:  # noqa: BLE001
        store.fail_step(meta.run_id, "activation_features", str(exc))
        print(f"\nERROR in Step 08 activation-features: {exc}", file=sys.stderr)
        return 1

    payload = {
        "model_ref": result.model_ref,
        "method": result.method,
        "calib_path": result.calib_path,
        "n_docs_used": result.n_docs_used,
        "n_tokens_est": result.n_tokens_est,
        "n_tensors": result.n_tensors,
        "n_with_features": result.n_with_features,
        "catalog_sha256": result.catalog_sha256,
        "hardest_groups": result.hardest_groups,
        "easiest_groups": result.easiest_groups,
        "steps_log": result.steps_log,
        "notes": result.notes,
    }
    log_text = "\n".join(result.steps_log) + "\n"

    store.complete_step(
        meta.run_id,
        "activation_features",
        payload,
        log_text=log_text,
        extra_artifacts={
            "tensor_catalog.json": json.dumps(updated, indent=2).encode("utf-8")
            + b"\n",
            "activation_features.json": json.dumps(
                {
                    "method": result.method,
                    "hardest_groups": result.hardest_groups,
                    "easiest_groups": result.easiest_groups,
                    "group_activation_features": updated.get(
                        "group_activation_features"
                    ),
                },
                indent=2,
            ).encode("utf-8")
            + b"\n",
        },
    )

    if print_explain:
        _explain_activation_features(result)
        print("\n=== checkpoint saved ===")
        print(f"  run_id   : {meta.run_id}")
        print(f"  step dir : {step_dir}")
        print(
            "  files    : output.json, tensor_catalog.json, "
            "activation_features.json, status.json, log.txt"
        )
        print("\n" + store.summary(meta.run_id))
        print("\n=== activation-features summary (JSON) ===")
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, indent=2))

    return 0


def cmd_freeze_gguf(args: argparse.Namespace) -> int:
    from odg.freeze import freeze_gguf
    from odg.store import StepAlreadyDone

    print_explain = not args.no_explain
    store = _store(args)

    try:
        meta = _require_run(store, model=args.model, run_id=args.run)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not store.is_step_done(meta.run_id, "activation_features"):
        print(
            "ERROR: Step 08 (activation-features) is not done.\n"
            f"  Run: odg activation-features --model {meta.model_ref}",
            file=sys.stderr,
        )
        return 1

    if print_explain:
        _banner_freeze_gguf(meta.model_ref, meta.run_id, meta.root)

    if store.is_step_done(meta.run_id, "freeze_gguf") and not args.force:
        out = store.read_step_output(meta.run_id, "freeze_gguf")
        if print_explain:
            print("Step 09 already checkpointed as done — loading from store.")
            print(f"  path: {store.step_path(meta.run_id, 'freeze_gguf')}")
            print("  (pass --force to re-run)\n")
            print(json.dumps(out, indent=2))
            print("\n" + store.summary(meta.run_id))
        elif out:
            print(json.dumps(out, indent=2))
        return 0

    resolve_out = store.read_step_output(meta.run_id, "resolve") or {}
    load_out = store.read_step_output(meta.run_id, "load") or {}

    source_path = load_out.get("source_path") or resolve_out.get("local_path")
    source_is_quantized = bool(
        load_out.get("source_is_quantized") or resolve_out.get("source_is_quantized")
    )

    hf_local = None
    local_path = resolve_out.get("local_path")
    if local_path and Path(local_path).is_dir():
        hf_local = local_path

    # Catalog names for verification (prefer activation-features catalog)
    catalog_names: list[str] = []
    for step_id in ("activation_features", "weight_features", "catalog"):
        cpath = store.step_path(meta.run_id, step_id) / "tensor_catalog.json"
        if cpath.is_file():
            cat = json.loads(cpath.read_text())
            catalog_names = list((cat.get("tensors") or {}).keys())
            break

    input_data = {
        "from_step": "activation_features",
        "mode": args.mode,
        "source_path": source_path,
        "hf_local_path": hf_local,
        "require_bf16": bool(args.require_bf16),
        "convert_script": str(args.convert_script) if args.convert_script else None,
        "n_catalog_tensors": len(catalog_names),
    }

    try:
        step_dir = store.begin_step(
            meta.run_id, "freeze_gguf", input_data, force=args.force
        )
    except StepAlreadyDone:
        return cmd_freeze_gguf(args)

    try:
        result = freeze_gguf(
            model_ref=meta.model_ref,
            out_dir=step_dir,
            source_path=source_path,
            source_is_quantized=source_is_quantized,
            hf_local_path=hf_local,
            catalog_tensor_names=catalog_names,
            mode=args.mode,
            convert_script=args.convert_script,
            require_bf16=bool(args.require_bf16),
        )
    except Exception as exc:  # noqa: BLE001
        store.fail_step(meta.run_id, "freeze_gguf", str(exc))
        print(f"\nERROR in Step 09 freeze-gguf: {exc}", file=sys.stderr)
        return 1

    payload = result.summary_dict()
    log_text = "\n".join(result.steps_log) + "\n"
    manifest = {
        **payload,
        "gguf_sha256_file": result.gguf_path + ".sha256",
    }

    # GGUF already in step_dir; record sha + manifest as extras
    store.complete_step(
        meta.run_id,
        "freeze_gguf",
        payload,
        log_text=log_text,
        extra_artifacts={
            "freeze_manifest.json": json.dumps(manifest, indent=2).encode("utf-8")
            + b"\n",
        },
    )

    if print_explain:
        _explain_freeze_gguf(result)
        print("\n=== checkpoint saved ===")
        print(f"  run_id   : {meta.run_id}")
        print(f"  step dir : {step_dir}")
        print(
            "  files    : output.json, model-*.gguf, *.sha256, "
            "freeze_manifest.json, status.json, log.txt"
        )
        print("\n" + store.summary(meta.run_id))
        print("\n=== freeze-gguf summary (JSON) ===")
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


def _banner_catalog(model: str, run_id: str, root: str) -> None:
    print(
        """
╔══════════════════════════════════════════════════════════════╗
║  OpenDynamicGGUF — Step 05: Build tensor catalog            ║
╚══════════════════════════════════════════════════════════════╝
""".rstrip()
    )
    print(
        f"""
What this step does
-------------------
  Input : Step 04 classified.json (+ resolve/load provenance)
  Model : {model}
  Goal  : Single source-of-truth tensor_catalog.json with
          gguf/hf names, roles, groups, and feature slots.

  Checkpoint
  ----------
  run_id : {run_id}
  root   : {root}
""".rstrip()
    )


def _explain_catalog(catalog) -> None:
    print("\n=== what happened ===")
    for line in catalog.steps_log:
        print(f"  • {line}")

    print("\n=== verdict ===")
    print(f"  ✓ Tensors            : {catalog.n_tensors}")
    print(f"  ✓ Groups             : {catalog.n_groups}")
    print(f"  ✓ Quantizable        : {catalog.n_quantizable}")
    print(f"  ✓ catalog_sha256     : {catalog.catalog_sha256[:24]}…")
    print(f"  ✓ source_sha256      : {(catalog.source_sha256 or '-')[:24]}…")
    print(f"  ✓ Backend            : {catalog.source_backend}")

    print("\n=== sample catalog entries ===")
    for i, (name, t) in enumerate(catalog.tensors.items()):
        if i >= 6:
            break
        print(f"  {name}")
        print(f"      role={t.role} depth={t.depth} group={t.group_id} Q={t.quantizable}")
        print(f"      hf_name={t.hf_name}")

    print("\n=== next ===")
    print("  Step 06 — compute weight features (fill weight_features slots).")


def _banner_weight_features(model: str, run_id: str, root: str) -> None:
    print(
        """
╔══════════════════════════════════════════════════════════════╗
║  OpenDynamicGGUF — Step 06: Weight features                 ║
╚══════════════════════════════════════════════════════════════╝
""".rstrip()
    )
    print(
        f"""
What this step does
-------------------
  Input : Step 05 tensor_catalog.json + GGUF weights
  Model : {model}
  Goal  : Cheap per-tensor stats (mean/var/sparsity/outliers/norms)
          to RANK which groups to probe first — not final bits.

  Checkpoint
  ----------
  run_id : {run_id}
  root   : {root}
""".rstrip()
    )


def _explain_weight_features(result) -> None:
    print("\n=== what happened ===")
    for line in result.steps_log:
        print(f"  • {line}")

    print("\n=== verdict ===")
    print(f"  ✓ Features filled   : {result.n_with_features}/{result.n_tensors}")
    print(f"  ✓ Skipped           : {result.n_skipped}")
    print(f"  ✓ catalog_sha256    : {result.catalog_sha256[:24]}…")
    print(f"  ✓ Source quantized  : {result.source_is_quantized}")

    print("\n=== hardest groups (probe these carefully) ===")
    for g in result.hardest_groups[:5]:
        print(
            f"  {g['group_id']:<22} hardness={g.get('hardness', 0):.4f} "
            f"outlier={g.get('outlier_ratio_mean', 0):.4f} "
            f"var={g.get('variance_mean', 0):.6f}"
        )

    print("\n=== easiest groups ===")
    for g in result.easiest_groups[:5]:
        print(
            f"  {g['group_id']:<22} hardness={g.get('hardness', 0):.4f} "
            f"sparsity={g.get('sparsity_mean', 0):.3f}"
        )

    if result.notes:
        print("\n=== notes ===")
        for n in result.notes:
            print(f"  • {n}")

    print("\n=== next ===")
    print("  Step 07 — build calibration corpus (calib / search / held-out).")


def _banner_corpus(model: str, run_id: str, root: str) -> None:
    print(
        """
╔══════════════════════════════════════════════════════════════╗
║  OpenDynamicGGUF — Step 07: Calibration corpus              ║
╚══════════════════════════════════════════════════════════════╝
""".rstrip()
    )
    print(
        f"""
What this step does
-------------------
  Input : Resolve descriptor (chat_template, specialty_domain)
  Model : {model}
  Goal  : Mixed-domain prompts → chat template → 60/20/20 split
          calib.txt / search.txt / heldout.txt

  Hard rule: heldout is for validation ONLY (never for search).

  Checkpoint
  ----------
  run_id : {run_id}
  root   : {root}
""".rstrip()
    )


def _explain_corpus(result) -> None:
    print("\n=== what happened ===")
    for line in result.steps_log:
        print(f"  • {line}")

    print("\n=== verdict ===")
    print(f"  ✓ Documents          : {result.n_documents}")
    print(
        f"  ✓ Split              : calib={result.n_calib} "
        f"search={result.n_search} heldout={result.n_heldout}"
    )
    print(
        f"  ✓ Tokens (est)       : total={result.tokens_est_total} "
        f"calib={result.tokens_est_calib} "
        f"search={result.tokens_est_search} "
        f"heldout={result.tokens_est_heldout}"
    )
    print(f"  ✓ Template           : {result.chat_template}")
    print(f"  ✓ Specialty          : {result.specialty_domain}")
    print(f"  ✓ Domains            : {result.domain_counts}")

    print("\n=== files ===")
    for k, p in result.files.items():
        print(f"  {k:<8} {p}")

    if result.notes:
        print("\n=== notes ===")
        for n in result.notes:
            print(f"  • {n}")

    print("\n=== next ===")
    print("  Step 08 — activation features (needs calib.txt + a forward pass).")


def _banner_activation_features(model: str, run_id: str, root: str) -> None:
    print(
        """
╔══════════════════════════════════════════════════════════════╗
║  OpenDynamicGGUF — Step 08: Activation features             ║
╚══════════════════════════════════════════════════════════════╝
""".rstrip()
    )
    print(
        f"""
What this step does
-------------------
  Input : calib.txt + catalog (weight_features) + optional BF16 HF model
  Model : {model}
  Goal  : Per-tensor activation range / outliers for probe ranking.
          Prefer real forward hooks; proxy_from_weights if no BF16.

  Checkpoint
  ----------
  run_id : {run_id}
  root   : {root}
""".rstrip()
    )


def _explain_activation_features(result) -> None:
    print("\n=== what happened ===")
    for line in result.steps_log:
        print(f"  • {line}")

    print("\n=== verdict ===")
    print(f"  ✓ Method            : {result.method}")
    print(f"  ✓ Features filled   : {result.n_with_features}/{result.n_tensors}")
    print(f"  ✓ Calib docs used   : {result.n_docs_used} (tokens_est≈{result.n_tokens_est})")
    print(f"  ✓ catalog_sha256    : {result.catalog_sha256[:24]}…")

    print("\n=== hardest groups (activation) ===")
    for g in result.hardest_groups[:5]:
        print(
            f"  {g['group_id']:<22} hardness={g.get('hardness', 0):.4f} "
            f"absmax={g.get('absmax', 0):.4f} "
            f"outlier={g.get('outlier_ratio', 0):.4f}"
        )

    print("\n=== easiest groups ===")
    for g in result.easiest_groups[:5]:
        print(
            f"  {g['group_id']:<22} hardness={g.get('hardness', 0):.4f} "
            f"absmax={g.get('absmax', 0):.4f}"
        )

    if result.notes:
        print("\n=== notes ===")
        for n in result.notes:
            print(f"  • {n}")

    print("\n=== next ===")
    print("  Step 09 — freeze BF16 GGUF for llama.cpp (imatrix / probes).")


def _banner_freeze_gguf(model: str, run_id: str, root: str) -> None:
    print(
        """
╔══════════════════════════════════════════════════════════════╗
║  OpenDynamicGGUF — Step 09: Freeze GGUF reference           ║
╚══════════════════════════════════════════════════════════════╝
""".rstrip()
    )
    print(
        f"""
What this step does
-------------------
  Input : HF BF16 dir (ideal) or working GGUF from resolve/load
  Model : {model}
  Goal  : One hashed GGUF file for imatrix / probes / export.
          Ideal: convert_hf_to_gguf.py --outtype bf16
          Now:   promote Ollama/Q8 GGUF if BF16 convert unavailable.

  Checkpoint
  ----------
  run_id : {run_id}
  root   : {root}
""".rstrip()
    )


def _explain_freeze_gguf(result) -> None:
    print("\n=== what happened ===")
    for line in result.steps_log:
        print(f"  • {line}")

    print("\n=== verdict ===")
    print(f"  ✓ Method            : {result.method}")
    print(f"  ✓ GGUF              : {result.gguf_path}")
    print(f"  ✓ sha256            : {result.gguf_sha256[:32]}…")
    print(f"  ✓ Size              : {result.gguf_nbytes / (1024**2):.1f} MiB")
    print(f"  ✓ BF16 reference    : {result.is_bf16_reference}")
    print(f"  ✓ dtypes            : {result.dtype_summary}")
    print(f"  ✓ Catalog match     : {result.catalog_match}")

    if result.notes:
        print("\n=== notes ===")
        for n in result.notes:
            print(f"  • {n}")

    print("\n=== next ===")
    print("  Step 10 — build imatrix from calib.txt + this frozen GGUF.")


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
