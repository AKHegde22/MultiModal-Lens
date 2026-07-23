"""Causal Path Patching for Vision-Language Models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import torch
from PIL import Image

from multimodallens.adapters.base import ModelAdapter
from multimodallens.core.hooks import ForwardHookCache, ForwardLayerPatcher, ForwardInputPatcher
from multimodallens.analysis.hooking import _forward_adapter, list_hookable_layers


@dataclass
class EdgeEffect:
    """Dataclass holding the effect of patching a single edge."""
    sender_layer: str
    sender_head: int | None      # None = MLP or whole layer
    receiver_layer: str  
    receiver_head: int | None    # None = MLP or whole layer
    receiver_channel: str        # 'q', 'k', 'v', or 'residual'
    clean_metric: float
    corrupt_metric: float
    patched_metric: float
    causal_effect: float         # normalized: (patched - corrupt) / (clean - corrupt)


@dataclass
class PathPatchEffect:
    """Backward compatibility alias."""
    sender_layer: str
    receiver_layer: str
    clean_score: float
    corrupted_score: float
    patched_score: float
    causal_effect: float


@dataclass
class PathPatchingResult:
    """Result of a causal path patching experiment."""
    model_family: str
    model_name: str
    prompt: str
    effects: list[EdgeEffect]

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "model_family": self.model_family,
            "model_name": self.model_name,
            "prompt": self.prompt,
            "effects": [
                {
                    "sender_layer": e.sender_layer,
                    "sender_head": e.sender_head,
                    "receiver_layer": e.receiver_layer,
                    "receiver_head": e.receiver_head,
                    "receiver_channel": e.receiver_channel,
                    "clean_metric": e.clean_metric,
                    "corrupt_metric": e.corrupt_metric,
                    "patched_metric": e.patched_metric,
                    "causal_effect": e.causal_effect,
                }
                for e in self.effects
            ],
        }

    def to_html(self) -> str:
        """Render interactive HTML dashboard string."""
        from multimodallens.viz.interactive import create_path_patching_html
        return create_path_patching_html(self)

    def to_plotly(self) -> Any:
        """Return Plotly Figure object."""
        from multimodallens.viz.interactive import create_path_patching_plotly_figure

        return create_path_patching_plotly_figure(self)

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
    receiver_channels: list[str] | None = None,
    metric_fn: Callable[[Any], float] = default_logit_metric,
) -> PathPatchingResult:
    """Evaluate causal path intervention from sender components to receiver components.
    
    This patches the specific edge between the sender layer and the receiver component's input.
    """
    adapter.ensure_loaded()
    assert adapter.model is not None

    clean_batch = adapter.prepare(clean_image, clean_prompt)
    corrupted_batch = adapter.prepare(corrupted_image, corrupted_prompt)

    all_hookable = list_hookable_layers(adapter)
    senders = sender_layers or all_hookable[: min(3, len(all_hookable))]
    receivers = receiver_layers or all_hookable[-min(3, len(all_hookable)) :]
    channels = receiver_channels or ["q", "k", "v"]

    # 1. Capture clean activations at sender layers
    with torch.no_grad():
        with ForwardHookCache(adapter.model, senders) as clean_cache:
            clean_out = _forward_adapter(adapter, clean_batch.model_inputs, requires_grad=False)
            clean_score = metric_fn(clean_out)

        # 2. Capture corrupt activations at sender layers and baseline corrupt score
        with ForwardHookCache(adapter.model, senders) as corrupt_cache:
            corrupted_out = _forward_adapter(adapter, corrupted_batch.model_inputs, requires_grad=False)
            corrupted_score = metric_fn(corrupted_out)

    effects: list[EdgeEffect] = []

    # 3. Perform path patching interventions
    with torch.no_grad():
        for s_layer in senders:
            clean_sender_out = clean_cache.activations.get(s_layer)
            corrupt_sender_out = corrupt_cache.activations.get(s_layer)
            
            if clean_sender_out is None or corrupt_sender_out is None:
                continue

            # This is (sender_clean_output - sender_corrupt_output)
            sender_diff = clean_sender_out - corrupt_sender_out

            for r_layer in receivers:
                for channel in channels:
                    if channel in ["q", "k", "v"]:
                        target_module_path = f"{r_layer}.self_attn.{channel}_proj"
                    else:
                        target_module_path = r_layer

                    # Verify module exists
                    module_dict = dict(adapter.model.named_modules())
                    if target_module_path not in module_dict:
                        continue

                    def patch_receiver_input(tensor: torch.Tensor) -> torch.Tensor:
                        diff = sender_diff.to(device=tensor.device, dtype=tensor.dtype)
                        return tensor + diff

                    # We patch the INPUT to the receiver module
                    with ForwardInputPatcher(adapter.model, target_module_path, patch_receiver_input):
                        patched_out = _forward_adapter(adapter, corrupted_batch.model_inputs, requires_grad=False)
                        patched_score = metric_fn(patched_out)

                    effect = (patched_score - corrupted_score) / (clean_score - corrupted_score + 1e-8)

                    effects.append(
                        EdgeEffect(
                            sender_layer=s_layer,
                            sender_head=None,
                            receiver_layer=r_layer,
                            receiver_head=None,
                            receiver_channel=channel,
                            clean_metric=clean_score,
                            corrupt_metric=corrupted_score,
                            patched_metric=patched_score,
                            causal_effect=float(effect),
                        )
                    )

    return PathPatchingResult(
        model_family=adapter.family,
        model_name=adapter.model_name,
        prompt=clean_prompt,
        effects=effects,
    )
