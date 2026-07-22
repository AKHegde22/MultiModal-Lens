"""Causal Path Patching for Vision-Language Models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import torch
from PIL import Image

from multimodallens.adapters.base import ModelAdapter
from multimodallens.core.hooks import ForwardHookCache, ForwardLayerPatcher
from multimodallens.analysis.hooking import _forward_adapter, list_hookable_layers


@dataclass
class PathPatchEffect:
    sender_layer: str
    receiver_layer: str
    clean_score: float
    corrupted_score: float
    patched_score: float
    causal_effect: float


@dataclass
class PathPatchingResult:
    model_family: str
    model_name: str
    prompt: str
    effects: list[PathPatchEffect]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_family": self.model_family,
            "model_name": self.model_name,
            "prompt": self.prompt,
            "effects": [
                {
                    "sender_layer": e.sender_layer,
                    "receiver_layer": e.receiver_layer,
                    "clean_score": e.clean_score,
                    "corrupted_score": e.corrupted_score,
                    "patched_score": e.patched_score,
                    "causal_effect": e.causal_effect,
                }
                for e in self.effects
            ],
        }

    def to_html(self) -> str:
        """Render interactive HTML dashboard string."""
        from multimodallens.viz.interactive import create_path_patching_html

        return create_path_patching_html(self)

    def save_html(self, filepath: str) -> None:
        """Save interactive HTML dashboard to disk."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_html())



def default_logit_metric(output: Any) -> float:
    """Default metric evaluating the max logit value from model outputs."""
    logits = getattr(output, "logits", None)
    if logits is None and isinstance(output, (tuple, list)) and len(output) > 0:
        logits = output[0]
    if torch.is_tensor(logits):
        return float(logits.max().item())
    return 0.0


def run_causal_path_patching(
    adapter: ModelAdapter,
    clean_image: Image.Image,
    clean_prompt: str,
    corrupted_image: Image.Image,
    corrupted_prompt: str,
    sender_layers: list[str] | None = None,
    receiver_layers: list[str] | None = None,
    metric_fn: Callable[[Any], float] = default_logit_metric,
) -> PathPatchingResult:
    """Evaluate causal path intervention from sender components to receiver components."""
    adapter.ensure_loaded()
    assert adapter.model is not None

    clean_batch = adapter.prepare(clean_image, clean_prompt)
    corrupted_batch = adapter.prepare(corrupted_image, corrupted_prompt)

    all_hookable = list_hookable_layers(adapter)
    senders = sender_layers or all_hookable[: min(3, len(all_hookable))]
    receivers = receiver_layers or all_hookable[-min(3, len(all_hookable)) :]

    # 1. Capture clean activations at sender layers
    with torch.no_grad():
        with ForwardHookCache(adapter.model, senders) as clean_cache:
            clean_out = _forward_adapter(adapter, clean_batch.model_inputs, requires_grad=False)
            clean_score = metric_fn(clean_out)

        # 2. Run corrupted forward baseline
        corrupted_out = _forward_adapter(adapter, corrupted_batch.model_inputs, requires_grad=False)
        corrupted_score = metric_fn(corrupted_out)

    effects: list[PathPatchEffect] = []

    # 3. Perform path patching interventions
    with torch.no_grad():
        for s_layer in senders:
            clean_activation = clean_cache.activations.get(s_layer)
            if clean_activation is None:
                continue

            for r_layer in receivers:

                def patch_sender(tensor: torch.Tensor) -> torch.Tensor:
                    return clean_activation.to(device=tensor.device, dtype=tensor.dtype)

                with ForwardLayerPatcher(adapter.model, s_layer, patch_sender):
                    patched_out = _forward_adapter(adapter, corrupted_batch.model_inputs, requires_grad=False)
                    patched_score = metric_fn(patched_out)

                effect = (patched_score - corrupted_score) / (clean_score - corrupted_score + 1e-8)

                effects.append(
                    PathPatchEffect(
                        sender_layer=s_layer,
                        receiver_layer=r_layer,
                        clean_score=clean_score,
                        corrupted_score=corrupted_score,
                        patched_score=patched_score,
                        causal_effect=float(effect),
                    )
                )

    return PathPatchingResult(
        model_family=adapter.family,
        model_name=adapter.model_name,
        prompt=clean_prompt,
        effects=effects,
    )
