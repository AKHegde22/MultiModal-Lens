"""Image utilities for overlay and perturbation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
import matplotlib

from multimodallens.utils.tensor_ops import safe_minmax


@dataclass(slots=True)
class PatchLayout:
    rows: int
    cols: int


def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    """Convert PIL image to uint8 RGB numpy array."""
    return np.array(image.convert("RGB"), dtype=np.uint8)


def overlay_heatmap(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "magma",
) -> np.ndarray:
    """Overlay normalized heatmap onto image.

    Args:
        image: PIL image.
        heatmap: 2D map in arbitrary range.
        alpha: Heatmap blending factor.
        colormap: Matplotlib colormap name.
    Returns:
        RGB uint8 array.
    """
    rgb = pil_to_rgb_array(image).astype(np.float32) / 255.0
    hm = safe_minmax(heatmap)

    hm_img = Image.fromarray((hm * 255).astype(np.uint8)).resize(
        (rgb.shape[1], rgb.shape[0]),
        resample=Image.Resampling.BICUBIC,
    )
    hm_resized = np.array(hm_img, dtype=np.float32) / 255.0

    cmap = matplotlib.colormaps.get_cmap(colormap)
    colored = cmap(hm_resized)[..., :3]

    blended = (1 - alpha) * rgb + alpha * colored
    blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
    return blended


def infer_patch_layout(num_patches: int) -> PatchLayout:
    """Infer near-square patch layout."""
    side = int(round(np.sqrt(num_patches)))
    if side * side != num_patches:
        rows = side
        cols = int(np.ceil(num_patches / max(rows, 1)))
        return PatchLayout(rows=rows, cols=cols)
    return PatchLayout(rows=side, cols=side)


def mask_top_patches(
    image: Image.Image,
    patch_scores: np.ndarray,
    patch_grid: tuple[int, int],
    mask_fraction: float,
    fill_mode: str = "mean",
) -> Image.Image:
    """Mask top-k scored patches in image.

    Args:
        image: Input PIL image.
        patch_scores: Flat patch scores [rows*cols] or grid [rows, cols].
        patch_grid: (rows, cols).
        mask_fraction: fraction in [0,1].
        fill_mode: mean|zero.
    Returns:
        Perturbed PIL image.
    """
    rows, cols = patch_grid
    arr = pil_to_rgb_array(image).copy()
    h, w, _ = arr.shape

    patch_h = max(h // rows, 1)
    patch_w = max(w // cols, 1)

    scores = patch_scores.reshape(rows, cols).reshape(-1)
    k = int(np.clip(mask_fraction, 0.0, 1.0) * scores.size)
    if k <= 0:
        return Image.fromarray(arr)

    indices = np.argsort(scores)[::-1][:k]

    if fill_mode == "mean":
        fill = arr.reshape(-1, 3).mean(axis=0).astype(np.uint8)
    else:
        fill = np.zeros((3,), dtype=np.uint8)

    for idx in indices:
        r = idx // cols
        c = idx % cols
        y0 = r * patch_h
        y1 = h if r == rows - 1 else min((r + 1) * patch_h, h)
        x0 = c * patch_w
        x1 = w if c == cols - 1 else min((c + 1) * patch_w, w)
        arr[y0:y1, x0:x1] = fill

    return Image.fromarray(arr)
