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
    "VisionLogitLensResult",
    "VisionLogitLensStep",
    "GroundingCircuitResult",
    "GroundingHeadScore",
    "HeadContribution",
    "MLPContribution",
    "EdgeEffect",
    "fold_layer_norms",
    "center_writing_weights",
    "center_unembed",
    "ForwardInputPatcher",
    "HookPoint",
    "LinearProbe",
    "evaluate_layer_probes",
    "detect_induction_heads",
    "detect_cross_modal_induction_heads",
    "SparseAutoencoder",
    "NeuronActivationSummary",
    "analyze_neuron_activations",
    "MultimodalLensError",
    "ModelLoadError",
    "UnsupportedFamilyError",
    "UnsupportedDtypeError",
    "AdapterError",
    "AnalysisError",
]

if TYPE_CHECKING:
    from multimodallens.adapters.generic_adapter import GenericVLMAdapter
    from multimodallens.analysis.dla import DLAResult, HeadContribution, MLPContribution
    from multimodallens.analysis.path_patching import PathPatchingResult, EdgeEffect
    from multimodallens.core.activation_cache import ActivationCache
    from multimodallens.core.config_schema import MultimodalConfig
    from multimodallens.core.hooked_vlm import HookedVLM
    from multimodallens.core.pipeline import LensPipeline
    from multimodallens.core.factored_matrix import FactoredMatrix
    from multimodallens.core.weight_processing import fold_layer_norms, center_writing_weights, center_unembed
    from multimodallens.core.hooks import ForwardInputPatcher
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
        VisionLogitLensResult,
        VisionLogitLensStep,
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
    if name in ("DLAResult", "HeadContribution", "MLPContribution"):
        import multimodallens.analysis.dla as dla

        return getattr(dla, name)
    if name in ("PathPatchingResult", "EdgeEffect"):
        import multimodallens.analysis.path_patching as path_patching

        return getattr(path_patching, name)
    if name == "FactoredMatrix":
        from multimodallens.core.factored_matrix import FactoredMatrix

        return FactoredMatrix
    if name in ("fold_layer_norms", "center_writing_weights", "center_unembed"):
        import multimodallens.core.weight_processing as weight_processing

        return getattr(weight_processing, name)
    if name in ("LinearProbe", "evaluate_layer_probes"):
        import multimodallens.analysis.probing as probing

        return getattr(probing, name)
    if name == "HookPoint":
        from multimodallens.core.hook_point import HookPoint

        return HookPoint
    if name == "SparseAutoencoder":
        from multimodallens.analysis.sae import SparseAutoencoder

        return SparseAutoencoder
    if name in ("detect_induction_heads", "detect_cross_modal_induction_heads"):
        import multimodallens.analysis.induction as induction

        return getattr(induction, name)
    if name in ("NeuronActivationSummary", "analyze_neuron_activations"):
        import multimodallens.analysis.neurons as neurons

        return getattr(neurons, name)
    if name == "ForwardInputPatcher":
        from multimodallens.core.hooks import ForwardInputPatcher

        return ForwardInputPatcher
    if name in (
        "AnalysisResult",
        "FaithfulnessMetrics",
        "LayerActivation",
        "LayerActivationRun",
        "ActivationPatchResult",
        "LogitLensResult",
        "LogitLensStep",
        "VisionLogitLensResult",
        "VisionLogitLensStep",
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

