"""Command line interface for MultimodalLens."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


def _cmd_ui(args: argparse.Namespace) -> int:
    from multimodallens.ui.app import build_app

    app = build_app()
    print(f"Launching MultimodalLens UI on http://{args.host}:{args.port}")
    app.queue(default_concurrency_limit=args.concurrency).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    from PIL import Image
    from multimodallens.core.pipeline import LensPipeline

    image_path = Path(args.image).expanduser()
    if not image_path.exists():
        print(f"Error: Image path '{image_path}' does not exist.", file=sys.stderr)
        return 1

    image = Image.open(image_path).convert("RGB")
    pipe = LensPipeline()

    print(f"Analyzing '{args.model}' (family: {args.family})...")
    result = pipe.analyze(
        family=args.family,
        model_name=args.model,
        image=image,
        prompt=args.prompt,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        compute_gradients=args.compute_gradients,
        run_faithfulness=args.run_faithfulness,
    )

    print("\n--- Analysis Result ---")
    print(f"Model: {result.model_name} ({result.model_family})")
    print(f"Global Score: {result.global_score:.6f}")
    print(f"Num Tokens: {len(result.tokens)}")
    print(f"Patch Grid: {result.patch_grid[0]} x {result.patch_grid[1]}")

    if result.faithfulness is not None:
        print("\n--- Faithfulness Metrics ---")
        print(f"Counterfactual Drop: {result.faithfulness.counterfactual_drop}")
        print(f"Attn-Grad Spearman: {result.faithfulness.attn_grad_spearman}")

    if args.output:
        import json
        out_path = Path(args.output).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model_family": result.model_family,
            "model_name": result.model_name,
            "prompt": result.prompt,
            "global_score": float(result.global_score),
            "patch_grid": list(result.patch_grid),
            "tokens": result.tokens,
            "token_scores": result.token_scores.tolist(),
        }
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"\nSaved analysis output to: {out_path}")

    return 0


def _cmd_preflight(args: argparse.Namespace) -> int:
    from multimodallens.core.pipeline import LensPipeline

    pipe = LensPipeline()
    print(f"Running preflight check for '{args.model}' (requested family: {args.family})...")
    report = pipe.preflight(
        family=args.family,
        model_name=args.model,
        trust_remote_code=args.trust_remote_code,
    )

    print("\n--- Preflight Report ---")
    print(f"Requested Family: {report.requested_family}")
    print(f"Resolved Family:  {report.resolved_family}")
    print(f"Model Type:       {report.model_type}")
    print(f"Architectures:    {', '.join(report.architectures)}")
    print(f"Vision Tower:     {'Yes' if report.has_vision_tower else 'No'}")
    print(f"Supports Explore: {'Yes' if report.supports_explore else 'No'}")

    if report.warnings:
        print("\nWarnings:")
        for w in report.warnings:
            print(f"  - {w}")

    if report.errors:
        print("\nErrors:", file=sys.stderr)
        for e in report.errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print("\nPreflight check PASSED.")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from multimodallens.core.pipeline import LensPipeline
    from multimodallens.eval.runner import EvaluationRunner

    dataset_path = Path(args.dataset).expanduser()
    if not dataset_path.exists():
        print(f"Error: Dataset file '{dataset_path}' not found.", file=sys.stderr)
        return 1

    pipe = LensPipeline()
    runner = EvaluationRunner(pipe)

    print(f"Running evaluation on '{dataset_path}' using '{args.model}'...")
    df = runner.run(
        dataset_path=dataset_path,
        family=args.family,
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        compute_gradients=args.compute_gradients,
        run_faithfulness=args.run_faithfulness,
    )

    print("\n--- Evaluation Summary ---")
    print(f"Evaluated Samples: {len(df)}")
    if not df.empty and "global_score" in df.columns:
        print(f"Mean Global Score: {df['global_score'].mean():.6f}")

    if args.output:
        out_path = Path(args.output).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved evaluation results to: {out_path}")

    return 0


def _cmd_version(_args: argparse.Namespace) -> int:
    from multimodallens import __version__

    print(f"multimodallens v{__version__}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="multimodallens",
        description="MultimodalLens CLI: Mechanistic interpretability for vision-language models",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # ui subcommand
    ui_parser = subparsers.add_parser("ui", help="Launch interactive Gradio application")
    ui_parser.add_argument("--host", default="0.0.0.0", help="Host IP address to bind (default: 0.0.0.0)")
    ui_parser.add_argument("--port", type=int, default=7860, help="Port to run server (default: 7860)")
    ui_parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    ui_parser.add_argument("--concurrency", type=int, default=2, help="Concurrency limit")

    # analyze subcommand
    analyze_parser = subparsers.add_parser("analyze", help="Run analysis on an image and prompt")
    analyze_parser.add_argument("--model", required=True, help="Hugging Face model checkpoint or path")
    analyze_parser.add_argument("--image", required=True, help="Path to input image file")
    analyze_parser.add_argument("--prompt", required=True, help="Text prompt")
    analyze_parser.add_argument("--family", default="auto", help="Model family (default: auto)")
    analyze_parser.add_argument("--device", default="auto", help="Runtime device (default: auto)")
    analyze_parser.add_argument("--dtype", default="float16", help="Precision (default: float16)")
    analyze_parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom code from HF hub")
    analyze_parser.add_argument("--compute-gradients", action="store_true", help="Compute gradient attributions")
    analyze_parser.add_argument("--run-faithfulness", action="store_true", help="Run perturbation faithfulness tests")
    analyze_parser.add_argument("--output", help="Optional JSON output filepath")

    # preflight subcommand
    preflight_parser = subparsers.add_parser("preflight", help="Run lightweight compatibility check on a model")
    preflight_parser.add_argument("--model", required=True, help="Hugging Face model checkpoint")
    preflight_parser.add_argument("--family", default="auto", help="Model family (default: auto)")
    preflight_parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom code from HF hub")

    # eval subcommand
    eval_parser = subparsers.add_parser("eval", help="Run batch evaluation over a dataset JSONL file")
    eval_parser.add_argument("--dataset", required=True, help="Path to JSONL dataset file")
    eval_parser.add_argument("--model", required=True, help="Hugging Face model checkpoint")
    eval_parser.add_argument("--family", default="auto", help="Model family (default: auto)")
    eval_parser.add_argument("--device", default="auto", help="Runtime device (default: auto)")
    eval_parser.add_argument("--dtype", default="float16", help="Precision (default: float16)")
    eval_parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom code from HF hub")
    eval_parser.add_argument("--compute-gradients", action="store_true", help="Compute gradient attributions")
    eval_parser.add_argument("--run-faithfulness", action="store_true", help="Run perturbation tests")
    eval_parser.add_argument("--output", help="CSV output filepath for results")

    # version subcommand
    subparsers.add_parser("version", help="Print version information")

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    commands = {
        "ui": _cmd_ui,
        "analyze": _cmd_analyze,
        "preflight": _cmd_preflight,
        "eval": _cmd_eval,
        "version": _cmd_version,
    }

    cmd_fn = commands.get(args.command)
    if cmd_fn:
        return cmd_fn(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
