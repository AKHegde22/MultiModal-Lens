"""MultimodalLens package."""

from __future__ import annotations

from typing import Any

__all__ = ["LensPipeline"]


def __getattr__(name: str) -> Any:
    """Lazy attribute loading to avoid importing heavy deps at package import time."""
    if name == "LensPipeline":
        from .core.pipeline import LensPipeline

        return LensPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
