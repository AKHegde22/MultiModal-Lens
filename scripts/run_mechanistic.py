#!/usr/bin/env python3
"""Run mechanistic probes for MultimodalLens from the command line."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import sys
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from multimodallens.core.pipeline import LensPipeline  # noqa: E402


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mechanistic probes for MMLens.")
    parser.add_argument("mode", choices=["layers", "cache", "patch", "logit-lens", "grounding"])
    parser.add_argument(
        "--family",
        required=True,
        help="Family label or alias (e.g., auto, clip, qwen2_vl, idefics2).",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument("--target-image", help="Required for patch mode.")
    parser.add_argument("--layer-name")
    parser.add_argument("--layer-index", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--logit-top-k", type=int, default=5)
    parser.add_argument("--mask-fraction", type=float, default=0.3)

    parser.add_argument("--output", required=True, help="JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipe = LensPipeline()

    image = Image.open(args.image).convert("RGB")

    common_kwargs = {
        "family": args.family,
        "model_name": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
    }

    if args.mode == "layers":
        result = pipe.list_layers(**common_kwargs)
    elif args.mode == "cache":
        result = pipe.capture_internals(
            **common_kwargs,
            image=image,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
        )
    elif args.mode == "patch":
        if not args.target_image:
            raise ValueError("--target-image is required for patch mode.")
        target = Image.open(args.target_image).convert("RGB")
        result = pipe.activation_patch(
            **common_kwargs,
            source_image=image,
            target_image=target,
            prompt=args.prompt,
            layer_name=args.layer_name,
            layer_index=args.layer_index,
            visual_only=True,
        )
    elif args.mode == "logit-lens":
        result = pipe.logit_lens(
            **common_kwargs,
            image=image,
            prompt=args.prompt,
            top_k=args.logit_top_k,
        )
    else:
        result = pipe.grounding_heads(
            **common_kwargs,
            image=image,
            prompt=args.prompt,
            mask_fraction=args.mask_fraction,
            top_k=args.top_k,
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_to_jsonable(result), indent=2), encoding="utf-8")
    print(f"Saved mechanistic output to {out_path}")


if __name__ == "__main__":
    main()