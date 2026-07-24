from __future__ import annotations

import numpy as np
from PIL import Image
import pytest

torch = pytest.importorskip("torch")

from tests.test_dla_correctness import _DeterministicAdapter
from multimodallens.core.hooked_vlm import HookedVLM
from multimodallens.core.activation_cache import ActivationCache
from multimodallens.analysis.probing import LinearProbe, evaluate_layer_probes
from multimodallens.analysis.induction import detect_induction_heads


def test_generate_and_generate_with_hooks():
    adapter = _DeterministicAdapter(model_name="toy", device="cpu", dtype="float32")
    adapter.load()
    vlm = HookedVLM(adapter)

    image = Image.fromarray(np.zeros((24, 24, 3), dtype=np.uint8))
    prompt = "cat"

    gen_text = vlm.generate(image, prompt, max_new_tokens=5)
    assert isinstance(gen_text, str)

    hook_points = vlm.list_hook_points()
    def zero_hook(t):
        return t * 0.0

    gen_patched = vlm.generate_with_hooks(image, prompt, max_new_tokens=5, fwd_hooks=[(hook_points[0], zero_hook)])
    assert isinstance(gen_patched, str)


def test_linear_probe_and_evaluate_layers():
    X = np.random.randn(20, 16)
    y = np.array([0]*10 + [1]*10)

    probe = LinearProbe(layer_name="layer.0")
    probe.fit(X, y)
    acc = probe.score(X, y)
    assert 0.0 <= acc <= 1.0

    cache1 = ActivationCache({"layer.0": torch.tensor(X[:10]), "layer.1": torch.tensor(X[:10])})
    cache2 = ActivationCache({"layer.0": torch.tensor(X[10:]), "layer.1": torch.tensor(X[10:])})

    layer_scores = evaluate_layer_probes([cache1, cache2], y)
    assert "layer.0" in layer_scores
    assert "layer.1" in layer_scores
    assert 0.0 <= layer_scores["layer.0"] <= 1.0


def test_induction_head_detection():
    # Construct mock attention weight tensor with elevated attention at (j+1) for repeated tokens
    attn_layer0 = torch.zeros((1, 2, 4, 4))
    # Head 0 has high weight at [3, 1] (when token 3 matches token 0, it attends to 0+1=1)
    attn_layer0[0, 0, 3, 1] = 0.95
    
    tokens = ["A", "B", "C", "A"]

    scores = detect_induction_heads([attn_layer0], tokens)
    assert (0, 0) in scores
    assert scores[(0, 0)] == pytest.approx(0.95, abs=1e-2)
