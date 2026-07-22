from multimodallens.core.hooks import discover_transformer_layers
from tests.test_mechanistic_features import _ToyAdapter


def test_discover_fine_grained_submodules():
    adapter = _ToyAdapter(model_name="toy")
    adapter.load()

    layers = discover_transformer_layers(adapter.model, include_submodules=True)
    assert len(layers) > 0
    has_submodule = any("proj" in l or "layers" in l for l in layers)
    assert has_submodule
