"""MultimodalLens: Interactive interpretability and debugging toolkit for vision-language models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "LensPipeline",
    "HookedVLM",
    "ActivationCache",
    "MultimodalConfig",
    "GenericVLMAdapter",
    "DLAResult",
    "PathPatchingResult",
    "AnalysisResult",
    "FaithfulnessMetrics",
    "LayerActivation",
    "LayerActivationRun",
    "ActivationPatchResult",
    "LogitLensResult",
    "LogitLensStep",
    "GroundingCircuitResult",
    "GroundingHeadScore",
    "MultimodalLensError",
    "ModelLoadError",
    "UnsupportedFamilyError",
    "UnsupportedDtypeError",
    "AdapterError",
    "AnalysisError",
]

if TYPE_CHECKING:
    from multimodallens.adapters.generic_adapter import GenericVLMAdapter
    from multimodallens.analysis.dla import DLAResult
    from multimodallens.analysis.path_patching import PathPatchingResult
    from multimodallens.core.activation_cache import ActivationCache
    from multimodallens.core.config_schema import MultimodalConfig
    from multimodallens.core.hooked_vlm import HookedVLM
    from multimodallens.core.pipeline import LensPipeline
    from multimodallens.exceptions import (
        AdapterError,
        AnalysisError,
        ModelLoadError,
        MultimodalLensError,
        UnsupportedDtypeError,
        UnsupportedFamilyError,
    )
    from multimodallens.types import (
        ActivationPatchResult,
        AnalysisResult,
        FaithfulnessMetrics,
        GroundingCircuitResult,
        GroundingHeadScore,
        LayerActivation,
        LayerActivationRun,
        LogitLensResult,
        LogitLensStep,
    )


def __getattr__(name: str) -> Any:
    """Lazy attribute loading to avoid importing heavy dependencies at import time."""
    if name == "LensPipeline":
        from multimodallens.core.pipeline import LensPipeline

        return LensPipeline
    if name == "HookedVLM":
        from multimodallens.core.hooked_vlm import HookedVLM

        return HookedVLM
    if name == "ActivationCache":
        from multimodallens.core.activation_cache import ActivationCache

        return ActivationCache
    if name == "MultimodalConfig":
        from multimodallens.core.config_schema import MultimodalConfig

        return MultimodalConfig
    if name == "GenericVLMAdapter":
        from multimodallens.adapters.generic_adapter import GenericVLMAdapter

        return GenericVLMAdapter
    if name == "DLAResult":
        from multimodallens.analysis.dla import DLAResult

        return DLAResult
    if name == "PathPatchingResult":
        from multimodallens.analysis.path_patching import PathPatchingResult

        return PathPatchingResult
    if name in (
        "AnalysisResult",
        "FaithfulnessMetrics",
        "LayerActivation",
        "LayerActivationRun",
        "ActivationPatchResult",
        "LogitLensResult",
        "LogitLensStep",
        "GroundingCircuitResult",
        "GroundingHeadScore",
    ):
        import multimodallens.types as types

        return getattr(types, name)
    if name in (
        "MultimodalLensError",
        "ModelLoadError",
        "UnsupportedFamilyError",
        "UnsupportedDtypeError",
        "AdapterError",
        "AnalysisError",
    ):
        import multimodallens.exceptions as exceptions

        return getattr(exceptions, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

