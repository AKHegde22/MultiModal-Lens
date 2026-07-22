"""Multimodal Direct Logit Attribution (DLA)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import torch
from PIL import Image

from multimodallens.adapters.base import ModelAdapter
from multimodallens.analysis.hooking import list_hookable_layers
from multimodallens.core.hooks import ForwardHookCache
from multimodallens.analysis.logit_lens import _get_unembedding, _forward_adapter


@dataclass
class DLAContribution:
    layer_name: str
    target_token: str
    contribution_score: float
    norm: float


@dataclass
class DLAResult:
    model_family: str
    model_name: str
    prompt: str
    target_token: str
    contributions: list[DLAContribution]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_family": self.model_family,
            "model_name": self.model_name,
            "prompt": self.prompt,
            "target_token": self.target_token,
            "contributions": [
                {
                    "layer_name": c.layer_name,
                    "target_token": c.target_token,
                    "contribution_score": c.contribution_score,
                    "norm": c.norm,
                }
                for c in self.contributions
            ],
        }

    def to_html(self) -> str:
        """Render interactive HTML dashboard string."""
        from multimodallens.viz.interactive import create_dla_waterfall_html

        return create_dla_waterfall_html(self)

    def save_html(self, filepath: str) -> None:
        """Save interactive HTML dashboard to disk."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_html())



def run_multimodal_dla(
    adapter: ModelAdapter,
    image: Image.Image,
    prompt: str,
    target_token: str | int,
    layer_names: list[str] | None = None,
    include_patterns: list[str] | None = None,
) -> DLAResult:
    """Compute Direct Logit Attribution (DLA) of intermediate layer outputs to target token logit."""
    adapter.ensure_loaded()
    assert adapter.model is not None

    unembed = _get_unembedding(adapter)
    if unembed is None:
        raise RuntimeError("Model does not expose output embeddings required for Direct Logit Attribution.")

    batch = adapter.prepare(image, prompt)
    selected = layer_names or list_hookable_layers(adapter, include_patterns=include_patterns)
    if not selected:
        raise RuntimeError("No hookable layers were found for DLA analysis.")

    # Target token id resolution
    target_id: int
    target_str: str
    if isinstance(target_token, int):
        target_id = target_token
        target_str = str(target_token)
    else:
        target_str = target_token
        tokenizer = getattr(adapter, "tokenizer", None)
        if tokenizer is None and hasattr(adapter, "processor"):
            tokenizer = getattr(adapter.processor, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "convert_tokens_to_ids"):
            target_id = int(tokenizer.convert_tokens_to_ids(target_token))
        else:
            target_id = 0

    with torch.no_grad():
        with ForwardHookCache(adapter.model, selected, max_tokens=None) as cache:
            _forward_adapter(adapter, batch.model_inputs, requires_grad=False)

    contributions: list[DLAContribution] = []

    # Get unembedding weights if possible
    unembed_weight = getattr(unembed, "weight", None)

    for layer_name in selected:
        act = cache.activations.get(layer_name)
        if act is None:
            continue

        if act.ndim == 3:
            h = act[0, -1].to(device=adapter.device, dtype=torch.float32)
        elif act.ndim == 2:
            h = act[-1].to(device=adapter.device, dtype=torch.float32)
        else:
            continue

        norm_val = float(torch.norm(h).item())

        if torch.is_tensor(unembed_weight) and target_id < unembed_weight.shape[0]:
            w = unembed_weight[target_id].to(device=h.device, dtype=torch.float32)
            score = float(torch.dot(h, w).item())
        else:
            try:
                logits = unembed(h.unsqueeze(0))
                score = float(logits[0, target_id].item())
            except Exception:
                score = 0.0

        contributions.append(
            DLAContribution(
                layer_name=layer_name,
                target_token=target_str,
                contribution_score=score,
                norm=norm_val,
            )
        )

    return DLAResult(
        model_family=adapter.family,
        model_name=adapter.model_name,
        prompt=prompt,
        target_token=target_str,
        contributions=contributions,
    )
