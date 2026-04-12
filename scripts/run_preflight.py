#!/usr/bin/env python3
"""Run model preflight checks for MultimodalLens compatibility."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from multimodallens.core.preflight import run_model_preflight  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight model preflight checks.")
    parser.add_argument("--family", required=True, help="Family label or alias (recommended: auto).")
    parser.add_argument("--model", required=True, help="Hugging Face model id")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output", help="Optional path to write JSON report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_model_preflight(
        family=args.family,
        model_name=args.model,
        trust_remote_code=args.trust_remote_code,
    )
    payload = asdict(report)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved preflight report to {out_path}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
