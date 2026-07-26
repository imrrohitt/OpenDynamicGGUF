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

    # --- status / runs ---
    p_status = sub.add_parser("status", help="Show checkpoint status for a run")
    p_status.add_argument("--run", default=None, help="run_id (default: latest for --model)")
    p_status.add_argument("--model", "-m", default=None)

    p_runs = sub.add_parser("runs", help="List all checkpointed runs")

    args = parser.parse_args(argv)

    if args.command == "resolve":
        return cmd_resolve(args)
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
