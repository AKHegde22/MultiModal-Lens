from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from multimodallens.core.activation_cache import ActivationCache  # noqa: E402


def test_activation_cache_dict_access():
    cache_dict = {
        "encoder.layers.0": torch.ones(1, 10, 16),
        "encoder.layers.1": torch.zeros(1, 10, 16),
    }
    cache = ActivationCache(cache_dict)

    assert len(cache) == 2
    assert "encoder.layers.0" in cache
    assert torch.equal(cache["encoder.layers.0"], torch.ones(1, 10, 16))
    assert cache.keys() == ["encoder.layers.0", "encoder.layers.1"]


def test_activation_cache_pattern_matching():
    cache_dict = {
        "vision_model.encoder.layers.5": torch.ones(1, 5),
        "language_model.model.layers.5": torch.zeros(1, 5),
    }
    cache = ActivationCache(cache_dict)

    # Exact match
    assert "vision_model.encoder.layers.5" in cache
    # Pattern match unique
    assert torch.equal(cache["*vision_model*"], torch.ones(1, 5))

    # Pattern match ambiguous
    with pytest.raises(KeyError, match="matched multiple layers"):
        _ = cache["*.layers.5"]

    # Non-existent
    with pytest.raises(KeyError, match="not found"):
        _ = cache["non_existent_layer"]


def test_activation_cache_to_device():
    cache_dict = {"layer_0": torch.ones(2, 2)}
    cache = ActivationCache(cache_dict)

    cpu_cache = cache.to("cpu")
    assert cpu_cache["layer_0"].device.type == "cpu"


def test_activation_cache_apply_to_cache():
    cache_dict = {"layer_0": torch.tensor([1.0, 2.0, 3.0])}
    cache = ActivationCache(cache_dict)

    doubled = cache.apply_to_cache(lambda x: x * 2.0)
    assert torch.equal(doubled["layer_0"], torch.tensor([2.0, 4.0, 6.0]))
