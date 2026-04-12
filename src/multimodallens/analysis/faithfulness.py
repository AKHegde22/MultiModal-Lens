"""Faithfulness tests for attribution maps."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import spearmanr

from multimodallens.types import FaithfulnessMetrics
from multimodallens.utils.image_ops import mask_top_patches

if TYPE_CHECKING:
    from PIL import Image
    from multimodallens.adapters.base import ModelAdapter


def perturbation_curves(
    adapter: "ModelAdapter",
    image: "Image.Image",
    prompt: str,
    patch_scores: np.ndarray,
    patch_grid: tuple[int, int],
    steps: int = 10,
) -> tuple[list[float], list[float]]:
    """Compute deletion and insertion curves.

    Deletion progressively removes top patches.
    Insertion progressively keeps top patches by masking complement.
    """
    deletion: list[float] = []
    insertion: list[float] = []

    base = adapter.score(image, prompt)
    fractions = np.linspace(0.0, 1.0, steps)

    for frac in fractions:
        deleted = mask_top_patches(image, patch_scores, patch_grid, float(frac), fill_mode="mean")
        del_score = adapter.score(deleted, prompt)
        deletion.append(float(base - del_score))

        complement_scores = -patch_scores
        inserted = mask_top_patches(image, complement_scores, patch_grid, float(frac), fill_mode="mean")
        ins_score = adapter.score(inserted, prompt)
        insertion.append(float(ins_score - base))

    return deletion, insertion


def counterfactual_drop(
    adapter: "ModelAdapter",
    image: "Image.Image",
    prompt: str,
    patch_scores: np.ndarray,
    patch_grid: tuple[int, int],
    fraction: float = 0.3,
) -> float:
    """Single-shot counterfactual masking score drop."""
    base = adapter.score(image, prompt)
    masked = mask_top_patches(image, patch_scores, patch_grid, fraction, fill_mode="mean")
    new_score = adapter.score(masked, prompt)
    return float(base - new_score)


def attention_gradient_agreement(attn_map: np.ndarray, grad_map: np.ndarray) -> float:
    """Rank correlation between attention and gradient maps."""
    a = attn_map.reshape(-1)
    g = grad_map.reshape(-1)

    # Spearman is undefined for constant inputs; treat as no measurable agreement.
    if a.size == 0 or g.size == 0:
        return 0.0
    if np.all(a == a[0]) or np.all(g == g[0]):
        return 0.0

    corr, _ = spearmanr(a, g)
    if np.isnan(corr):
        return 0.0
    return float(corr)


def build_faithfulness_metrics(
    adapter: "ModelAdapter",
    image: "Image.Image",
    prompt: str,
    patch_scores: np.ndarray,
    patch_grid: tuple[int, int],
    grad_map: np.ndarray | None = None,
) -> FaithfulnessMetrics:
    """Run all supported faithfulness diagnostics."""
    deletion, insertion = perturbation_curves(
        adapter=adapter,
        image=image,
        prompt=prompt,
        patch_scores=patch_scores,
        patch_grid=patch_grid,
    )

    drop = counterfactual_drop(
        adapter=adapter,
        image=image,
        prompt=prompt,
        patch_scores=patch_scores,
        patch_grid=patch_grid,
    )

    agreement = None
    if grad_map is not None:
        agreement = attention_gradient_agreement(patch_scores.reshape(patch_grid), grad_map)

    return FaithfulnessMetrics(
        deletion_curve=deletion,
        insertion_curve=insertion,
        counterfactual_drop=drop,
        attn_grad_spearman=agreement,
    )
