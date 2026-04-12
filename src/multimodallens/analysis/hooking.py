"""Layer-wise activation capture built on forward hooks."""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from multimodallens.adapters.base import ModelAdapter
from multimodallens.core.hooks import ForwardHookCache, discover_transformer_layers
from multimodallens.types import LayerActivation, LayerActivationRun
from multimodallens.utils.tensor_ops import to_numpy


def _forward_adapter(adapter: ModelAdapter, model_inputs: dict[str, Any], requires_grad: bool = False) -> Any:
    forward_fn = getattr(adapter, "_forward", None)
    if not callable(forward_fn):
        raise RuntimeError(
            f"Adapter '{adapter.__class__.__name__}' does not expose a callable _forward method."
        )
    return forward_fn(model_inputs, requires_grad=requires_grad)


def list_hookable_layers(
    adapter: ModelAdapter,
    include_patterns: list[str] | None = None,
) -> list[str]:
    """List transformer block module paths that can be hooked."""
    adapter.ensure_loaded()
    assert adapter.model is not None
    return discover_transformer_layers(adapter.model, include_patterns=include_patterns)


def capture_layer_activations(
    adapter: ModelAdapter,
    image: Image.Image,
    prompt: str,
    layer_names: list[str] | None = None,
    include_patterns: list[str] | None = None,
    max_tokens: int | None = 256,
) -> LayerActivationRun:
    """Capture per-layer hidden vectors for one input using forward hooks."""
    adapter.ensure_loaded()
    assert adapter.model is not None

    batch = adapter.prepare(image, prompt)
    selected = layer_names or discover_transformer_layers(adapter.model, include_patterns=include_patterns)
    if not selected:
        raise RuntimeError("No hookable transformer layers were discovered for this model.")

    with torch.no_grad():
        with ForwardHookCache(adapter.model, selected, max_tokens=max_tokens) as cache:
            _forward_adapter(adapter, batch.model_inputs, requires_grad=False)

    layers: list[LayerActivation] = []
    for name in selected:
        tensor = cache.activations.get(name)
        if tensor is None:
            continue
        layers.append(
            LayerActivation(
                layer_name=name,
                shape=tuple(int(d) for d in tensor.shape),
                values=to_numpy(tensor),
            )
        )

    return LayerActivationRun(
        model_family=adapter.family,
        model_name=adapter.model_name,
        prompt=prompt,
        layers=layers,
        tokens=batch.tokens,
    )