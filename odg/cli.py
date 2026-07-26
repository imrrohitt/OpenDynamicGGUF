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
        help="Also download the HF BF16 safetensors snapshot (needs Hub auth if gated)",
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
║  OpenDynamicGGUF — Step 01: Resolve model to original BF16  ║
╚══════════════════════════════════════════════════════════════╝
""".rstrip()
    )
    print(
        f"""
What this step does
-------------------
  Input : {model}
  Goal  : Find the ORIGINAL full-precision (BF16/F16) weights.

  Why   : Ollama/MLX copies are often already quantized (Q4/Q8/…).
          Re-quantizing them stacks error. We always go back to HF BF16.

  This step does NOT load the neural net and does NOT quantize.
""".rstrip()
    )


def _explain(result) -> None:
    print("\n=== what happened ===")
    for line in result.steps_log:
        print(f"  • {line}")

    print("\n=== verdict ===")
    if result.rejected_quantized_source:
        print("  ✗ Rejected quantized local artifact as a quantization source.")
        print(f"    {result.rejected_quantized_source}")
    print(f"  ✓ Kind           : {result.kind.value}")
    print(f"  ✓ Upstream HF    : {result.hf_repo_id}")
    print(f"  ✓ Local path     : {result.local_path}")
    print(f"  ✓ Weights ready  : {result.weights_ready}")

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
    if not result.weights_ready:
        print(
            "  BF16 weights are not on disk yet (common for gated Gemma models).\n"
            "  1. Accept the license on the HF model page\n"
            "  2. huggingface-cli login\n"
            "  3. odg resolve --model "
            f"{result.user_ref} --download-weights\n"
            "  Then continue to Step 02 (load model)."
        )
    else:
        print("  Weights ready → proceed to Step 02 (load model into memory).")


if __name__ == "__main__":
    raise SystemExit(main())
