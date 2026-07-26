"""CLI for OpenDynamicGGUF. Step 01: ``odg resolve``."""

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
    sub = parser.add_subparsers(dest="command", required=True)

    p_resolve = sub.add_parser(
        "resolve",
        help="Step 01: resolve any model ref to the original BF16 source",
    )
    p_resolve.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model ref: functiongemma:latest | google/... | ./local-dir | mlx id",
    )
    p_resolve.add_argument(
        "--download-weights",
        action="store_true",
        help="With --prefer-hf: download the HF BF16 safetensors snapshot",
    )
    p_resolve.add_argument(
        "--prefer-hf",
        action="store_true",
        help="For Ollama tags: use Hugging Face BF16 instead of the local Ollama GGUF",
    )
    p_resolve.add_argument(
        "--from-ollama",
        action="store_true",
        default=True,
        help="For Ollama tags: use local Ollama GGUF (default)",
    )
    p_resolve.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write resolve result JSON to this path (default: stdout only)",
    )
    p_resolve.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory for HF downloads (default: ~/.cache/odg/models)",
    )
    p_resolve.add_argument(
        "--explain",
        action="store_true",
        default=True,
        help="Print a human explanation of each step (default: on)",
    )
    p_resolve.add_argument(
        "--no-explain",
        action="store_true",
        help="Only print JSON result",
    )

    args = parser.parse_args(argv)

    if args.command == "resolve":
        return cmd_resolve(args)

    parser.error(f"Unknown command {args.command}")
    return 2


def cmd_resolve(args: argparse.Namespace) -> int:
    from odg.resolve import resolve_model

    print_explain = not args.no_explain

    if print_explain:
        _banner(args.model)

    try:
        result = resolve_model(
            args.model,
            cache_dir=args.cache_dir,
            download_weights=args.download_weights,
            prefer_hf=args.prefer_hf,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"\nERROR in Step 01 resolve: {exc}", file=sys.stderr)
        return 1

    if print_explain:
        _explain(result)

    payload = result.to_dict()
    text = json.dumps(payload, indent=2)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        if print_explain:
            print(f"\nWrote JSON → {args.out}")

    if args.no_explain:
        print(text)
    else:
        print("\n=== resolve result (JSON) ===")
        print(text)

    # Exit 0 even if weights not downloaded — identity resolution succeeded.
    # Exit 2 only if we somehow lack an HF id / local path after resolve.
    if not result.hf_repo_id and not result.local_path:
        return 2
    return 0


def _banner(model: str) -> None:
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
