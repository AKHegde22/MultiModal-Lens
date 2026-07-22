"""Custom exception hierarchy for MultimodalLens."""

from __future__ import annotations


class MultimodalLensError(Exception):
    """Base exception for all MultimodalLens errors."""


class ModelLoadError(MultimodalLensError):
    """Raised when a model or processor fails to download or initialize."""


class UnsupportedFamilyError(MultimodalLensError):
    """Raised when an unrecognized or unsupported model family is requested."""


class UnsupportedDtypeError(MultimodalLensError):
    """Raised when an invalid dtype string is provided."""


class AdapterError(MultimodalLensError):
    """Base exception for errors originating inside a model adapter."""


class AnalysisError(MultimodalLensError):
    """Raised when analysis processing fails at runtime."""
