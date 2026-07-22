from __future__ import annotations

import numpy as np
from PIL import Image
import pytest

torch = pytest.importorskip("torch")

from tests.test_mechanistic_features import _ToyAdapter  # noqa: E402
from multimodallens.core.hooked_vlm import HookedVLM  # noqa: E402


def test_hooked_vlm_properties():
    adapter = _ToyAdapter(model_name="toy", device="cpu", dtype="float32")
    vlm = HookedVLM(adapter)

    assert vlm.model_name == "toy"
    assert vlm.family == "llava"
    assert vlm.device.type == "cpu"
    assert vlm.model is not None


def test_hooked_vlm_list_hook_points():
    adapter = _ToyAdapter(model_name="toy", device="cpu", dtype="float32")
    vlm = HookedVLM(adapter)

    hook_points = vlm.list_hook_points()
    assert len(hook_points) >= 4
    assert any("layers.0" in h for h in hook_points)


def test_hooked_vlm_run_with_cache():
    adapter = _ToyAdapter(model_name="toy", device="cpu", dtype="float32")
    vlm = HookedVLM(adapter)

    image = Image.fromarray(np.zeros((24, 24, 3), dtype=np.uint8))
    result, cache = vlm.run_with_cache(image, "test prompt")

    assert result.global_score is not None
    assert len(cache) >= 4
    assert any("layers.0" in k for k in cache.keys())



def test_hooked_vlm_run_with_hooks():
    adapter = _ToyAdapter(model_name="toy", device="cpu", dtype="float32")
    vlm = HookedVLM(adapter)

    image = Image.fromarray(np.zeros((24, 24, 3), dtype=np.uint8))
    baseline = vlm.analyze(image, "test prompt")

    hook_points = vlm.list_hook_points()

    def zero_hook(tensor):
        return tensor * 0.0

    patched = vlm.run_with_hooks(image, "test prompt", fwd_hooks=[(hook_points[0], zero_hook)])
    assert patched.global_score != baseline.global_score
