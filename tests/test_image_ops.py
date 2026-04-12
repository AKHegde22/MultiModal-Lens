from __future__ import annotations

import numpy as np
from PIL import Image

from multimodallens.utils.image_ops import mask_top_patches, overlay_heatmap


def test_overlay_heatmap_shape():
    img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    hm = np.random.rand(4, 4)
    out = overlay_heatmap(img, hm)
    assert out.shape == (32, 32, 3)


def test_mask_top_patches_returns_pil():
    img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    scores = np.random.rand(16)
    out = mask_top_patches(img, scores, (4, 4), 0.5)
    assert isinstance(out, Image.Image)
