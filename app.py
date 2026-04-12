"""Application entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from multimodallens.ui.app import build_app  # noqa: E402


if __name__ == "__main__":
    app = build_app()
    app.queue(default_concurrency_limit=2).launch(server_name="0.0.0.0", server_port=7860)
