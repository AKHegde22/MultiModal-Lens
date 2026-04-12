"""Cross-modal activation patching for causal tracing."""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from multimodallens.adapters.base import ModelAdapter
from multimodallens.analysis.hooking import list_hookable_layers
from multimodallens.core.hooks import ForwardHookCache, ForwardLayerPatcher
from multimodallens.types import ActivationPatchResult


def _forward_adapter(adapter: ModelAdapter, model_inputs: dict[str, Any], requires_grad: bool = False) -> Any:
    forward_fn = getattr(adapter, "_forward", None)
    if not callable(forward_fn):
        raise RuntimeError(
            f"Adapter '{adapter.__class__.__name__}' does not expose a callable _forward method."
        )
    return forward_fn(model_inputs, requires_grad=requires_grad)


def _score_from_outputs(outputs: Any, model_inputs: dict[str, Any]) -> float:
    logits_per_image = getattr(outputs, "logits_per_image", None)
    if logits_per_image is not None:
        return float(logits_per_image[0, 0].item())

    logits = getattr(outputs, "logits", None)
    if logits is None:
        raise RuntimeError("Forward outputs do not expose logits for scoring.")

    input_ids = model_inputs.get("input_ids")
    if input_ids is None or logits.shape[1] < 2:
        return float(logits[0, -1].max().item())

    labels = input_ids[:, 1:]
    pred = logits[:, : labels.shape[1], :]
    log_probs = pred.log_softmax(dim=-1)
    token_logp = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return float(token_logp.mean().item())


def _choose_layer(
    adapter: ModelAdapter,
    layer_name: str | None,
    layer_index: int,
    include_patterns: list[str] | None,
) -> tuple[str, int, int]:
    names = list_hookable_layers(adapter, include_patterns=include_patterns)
    if not names:
        raise RuntimeError("No hookable layers were discovered for activation patching.")

    if layer_name is not None:
        if layer_name not in names:
            raise ValueError(f"Requested layer '{layer_name}' is not hookable for this model.")
        idx = names.index(layer_name)
        return layer_name, idx, len(names)

    idx = max(0, min(layer_index, len(names) - 1))
    return names[idx], idx, len(names)


def _infer_llava_visual_indices(adapter: ModelAdapter, batch: Any, outputs: Any) -> list[int]:
    segment_fn = getattr(adapter, "_segment_hidden_states", None)
    if not callable(segment_fn):
        return []
    try:
        _text_hidden, _image_hidden, _text_tokens, image_indices = segment_fn(outputs, batch)
    except Exception:
        return []
    return [int(i) for i in image_indices]


def run_cross_modal_activation_patch(
    adapter: ModelAdapter,
    source_image: Image.Image,
    target_image: Image.Image,
    prompt: str,
    layer_name: str | None = None,
    layer_index: int = 12,
    include_patterns: list[str] | None = None,
    visual_only: bool = True,
) -> ActivationPatchResult:
    """Swap a layer activation from source image into target image forward pass."""
    adapter.ensure_loaded()
    assert adapter.model is not None

    chosen_layer, chosen_index, layer_count = _choose_layer(
        adapter=adapter,
        layer_name=layer_name,
        layer_index=layer_index,
        include_patterns=include_patterns,
    )

    src_batch = adapter.prepare(source_image, prompt)
    tgt_batch = adapter.prepare(target_image, prompt)

    with torch.no_grad():
        tgt_outputs = _forward_adapter(adapter, tgt_batch.model_inputs, requires_grad=False)
        baseline_score = _score_from_outputs(tgt_outputs, tgt_batch.model_inputs)

    visual_positions: list[int] = []
    if visual_only and adapter.family == "llava":
        visual_positions = _infer_llava_visual_indices(adapter, tgt_batch, tgt_outputs)

    with torch.no_grad():
        with ForwardHookCache(adapter.model, [chosen_layer], max_tokens=None) as src_cache:
            _forward_adapter(adapter, src_batch.model_inputs, requires_grad=False)

    source_tensor = src_cache.activations.get(chosen_layer)
    if source_tensor is None:
        raise RuntimeError(f"Failed to capture source activation at layer '{chosen_layer}'.")

    patched_fraction = 1.0

    def _patch_fn(target_tensor: torch.Tensor) -> torch.Tensor:
        nonlocal patched_fraction

        src = source_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype)
        if src.shape != target_tensor.shape:
            raise RuntimeError(
                "Source and target layer activations have mismatched shapes: "
                f"source={tuple(src.shape)} target={tuple(target_tensor.shape)}"
            )

        if not visual_only or not visual_positions:
            patched_fraction = 1.0
            return src

        patched = target_tensor.clone()
        if target_tensor.ndim == 3:
            seq_len = int(target_tensor.shape[1])
            idx = [i for i in visual_positions if 0 <= i < seq_len]
            if not idx:
                patched_fraction = 0.0
                return patched
            patched[:, idx, :] = src[:, idx, :]
            patched_fraction = float(len(idx) / max(seq_len, 1))
            return patched

        if target_tensor.ndim == 2:
            seq_len = int(target_tensor.shape[0])
            idx = [i for i in visual_positions if 0 <= i < seq_len]
            if not idx:
                patched_fraction = 0.0
                return patched
            patched[idx, :] = src[idx, :]
            patched_fraction = float(len(idx) / max(seq_len, 1))
            return patched

        patched_fraction = 1.0
        return src

    with torch.no_grad():
        with ForwardLayerPatcher(adapter.model, chosen_layer, patch_fn=_patch_fn):
            patched_outputs = _forward_adapter(adapter, tgt_batch.model_inputs, requires_grad=False)
            patched_score = _score_from_outputs(patched_outputs, tgt_batch.model_inputs)

    return ActivationPatchResult(
        layer_name=chosen_layer,
        baseline_score=baseline_score,
        patched_score=patched_score,
        delta_score=patched_score - baseline_score,
        patched_fraction=patched_fraction,
        visual_only=visual_only,
        metadata={
            "layer_index": chosen_index,
            "total_hookable_layers": layer_count,
            "visual_positions": visual_positions,
        },
    )