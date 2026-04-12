"""Shared datatypes for pipeline I/O."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class FaithfulnessMetrics:
    """Faithfulness diagnostics derived from perturbation tests."""

    deletion_curve: list[float] = field(default_factory=list)
    insertion_curve: list[float] = field(default_factory=list)
    counterfactual_drop: float | None = None
    attn_grad_spearman: float | None = None


@dataclass(slots=True)
class AnalysisResult:
    """Normalized output produced by any adapter."""

    model_family: str
    model_name: str
    prompt: str
    tokens: list[str]
    image_size: tuple[int, int]
    patch_grid: tuple[int, int]
    global_score: float
    token_scores: np.ndarray
    alignment_matrix: np.ndarray
    attention_maps: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)
    faithfulness: FaithfulnessMetrics | None = None


@dataclass(slots=True)
class AdapterBatch:
    """Prepared model inputs and token metadata."""

    model_inputs: dict[str, Any]
    tokens: list[str]
    token_ids: list[int]


@dataclass(slots=True)
class LayerActivation:
    """Captured activation tensor for a single hooked layer."""

    layer_name: str
    shape: tuple[int, ...]
    values: np.ndarray


@dataclass(slots=True)
class LayerActivationRun:
    """Activation cache collected across multiple layers for a single input."""

    model_family: str
    model_name: str
    prompt: str
    layers: list[LayerActivation]
    tokens: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ActivationPatchResult:
    """Result of cross-example activation patching at a selected layer."""

    layer_name: str
    baseline_score: float
    patched_score: float
    delta_score: float
    patched_fraction: float
    visual_only: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LogitLensStep:
    """Top-token lens readout for one layer and sequence position."""

    layer_name: str
    position: int
    top_tokens: list[str]
    top_probabilities: list[float]


@dataclass(slots=True)
class LogitLensResult:
    """Layer-wise token predictions from intermediate hidden states."""

    model_family: str
    model_name: str
    prompt: str
    steps: list[LogitLensStep]


@dataclass(slots=True)
class GroundingHeadScore:
    """Per-head visual grounding sensitivity score."""

    layer_index: int
    head_index: int
    baseline_visual_mass: float
    ablated_visual_mass: float
    delta_visual_mass: float
    grounding_score: float


@dataclass(slots=True)
class GroundingCircuitResult:
    """Grounding-head discovery output for one model/input pair."""

    model_family: str
    model_name: str
    prompt: str
    mask_fraction: float
    baseline_score: float
    ablated_score: float
    score_drop: float
    heads: list[GroundingHeadScore]
