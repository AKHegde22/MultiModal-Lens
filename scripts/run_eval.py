#!/usr/bin/env python3
"""Run batch evaluation for MultimodalLens."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from multimodallens.eval.runner import EvaluationRunner  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MultimodalLens evaluation on a JSONL dataset.")
    parser.add_argument("--dataset", required=True, help="Path to JSONL with fields: image, text, optional id")
    parser.add_argument(
        "--family",
        required=True,
        help="Family label or alias (e.g., auto, clip, qwen2_vl, idefics2).",
    )
    parser.add_argument("--model", required=True, help="Hugging Face model id")
    parser.add_argument("--device", default="auto", help="cpu|cuda|mps|auto")
    parser.add_argument("--dtype", default="float16", help="float32|float16|bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--compute-gradients", action="store_true")
    parser.add_argument("--run-faithfulness", action="store_true")
    parser.add_argument("--output", default="eval_results.csv", help="CSV output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = EvaluationRunner()
    df = runner.run(
        dataset_path=args.dataset,
        family=args.family,
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        compute_gradients=args.compute_gradients,
        run_faithfulness=args.run_faithfulness,
    )
    out_path = Path(args.output)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
