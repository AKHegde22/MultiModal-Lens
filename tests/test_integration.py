from __future__ import annotations

import numpy as np
from PIL import Image
import pytest

torch = pytest.importorskip("torch")


@pytest.mark.slow
def test_full_pipeline_with_clip():
    from multimodallens import HookedVLM, LensPipeline

    # 1. Test HookedVLM
    vlm = HookedVLM.from_pretrained("openai/clip-vit-base-patch32", device="cpu", dtype="float32")
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    res, cache = vlm.run_with_cache(image, "a photo of a cat")
    assert res.global_score is not None
    assert len(cache) > 0

    # 2. Test LensPipeline logit lens
    pipe = LensPipeline()
    lens = pipe.logit_lens(model_name="openai/clip-vit-base-patch32", image=image, prompt="a cat", device="cpu", dtype="float32")
    assert lens.steps
