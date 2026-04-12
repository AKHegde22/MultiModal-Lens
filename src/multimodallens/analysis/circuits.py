"""Vision-text circuit discovery utilities."""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from multimodallens.adapters.base import ModelAdapter
from multimodallens.types import GroundingCircuitResult, GroundingHeadScore
from multimodallens.utils.image_ops import mask_top_patches


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


def _tensor_attentions(attentions: Any) -> list[torch.Tensor]:
    if attentions is None:
        return []
    if not isinstance(attentions, (tuple, list)):
        return []
    return [a.detach() for a in attentions if torch.is_tensor(a)]


def _extract_attention_stack(adapter: ModelAdapter, outputs: Any) -> list[torch.Tensor]:
    if adapter.family == "clip":
        vision = getattr(outputs, "vision_model_output", None)
        vis_attn = getattr(vision, "attentions", None) if vision is not None else None
        tensors = _tensor_attentions(vis_attn)
        if tensors:
            return tensors

    if adapter.family == "blip2":
        qformer = getattr(outputs, "qformer_outputs", None)
        cross = getattr(qformer, "cross_attentions", None) if qformer is not None else None
        tensors = _tensor_attentions(cross)
        if tensors:
            return tensors

    attentions = getattr(outputs, "attentions", None)
    tensors = _tensor_attentions(attentions)
    if tensors:
        return tensors

    if adapter.family == "clip":
        raise RuntimeError(
            "Model outputs do not expose attention tensors for circuit discovery. "
            "For CLIP-family models, use eager attention backend."
        )

    raise RuntimeError("Model outputs do not expose attention tensors for circuit discovery.")


def _llava_visual_indices(adapter: ModelAdapter, batch: Any, outputs: Any) -> list[int]:
    segment_fn = getattr(adapter, "_segment_hidden_states", None)
    if not callable(segment_fn):
        return []
    try:
        _text_hidden, _image_hidden, _text_tokens, image_indices = segment_fn(outputs, batch)
    except Exception:
        return []
    return [int(i) for i in image_indices]


def _build_visual_key_mask(
    adapter: ModelAdapter,
    batch: Any,
    outputs: Any,
    key_len: int,
) -> torch.Tensor:
    if adapter.family == "clip":
        mask = torch.zeros((key_len,), dtype=torch.bool)
        if key_len <= 1:
            mask[:] = True
        else:
            # CLIP vision attention uses position 0 as CLS and remaining positions as patches.
            mask[1:] = True
        return mask

    if adapter.family == "blip2":
        return torch.ones((key_len,), dtype=torch.bool)

    if adapter.family == "llava":
        indices = _llava_visual_indices(adapter, batch, outputs)
        mask = torch.zeros((key_len,), dtype=torch.bool)
        for idx in indices:
            if 0 <= idx < key_len:
                mask[idx] = True
        if int(mask.sum()) == 0:
            # Fall back to all keys so the feature remains usable across checkpoint variants.
            return torch.ones((key_len,), dtype=torch.bool)
        return mask

    # Generic fallback for future adapters: treat all keys as candidate visual keys.
    return torch.ones((key_len,), dtype=torch.bool)


def _head_visual_mass(attn: torch.Tensor, visual_mask: torch.Tensor) -> torch.Tensor:
    key_mask = visual_mask.to(device=attn.device)
    if key_mask.shape[0] != attn.shape[-1]:
        raise RuntimeError(
            f"Visual mask length {key_mask.shape[0]} does not match attention key length {attn.shape[-1]}."
        )

    if int(key_mask.sum()) == 0:
        return torch.zeros((attn.shape[1],), dtype=attn.dtype, device=attn.device)

    masked = attn[..., key_mask]
    return masked.mean(dim=(-1, -2))[0]


def discover_grounding_heads(
    adapter: ModelAdapter,
    image: Image.Image,
    prompt: str,
    mask_fraction: float = 0.3,
    top_k: int = 20,
) -> GroundingCircuitResult:
    """Rank attention heads by visual grounding sensitivity under ablation."""
    adapter.ensure_loaded()

    baseline_analysis = adapter.analyze(image=image, prompt=prompt, compute_gradients=False)
    if "vision_rollout" not in baseline_analysis.attention_maps:
        raise RuntimeError("Missing vision rollout map required for grounding-head discovery.")

    patch_scores = baseline_analysis.attention_maps["vision_rollout"].reshape(-1)
    ablated_image = mask_top_patches(
        image=image,
        patch_scores=patch_scores,
        patch_grid=baseline_analysis.patch_grid,
        mask_fraction=mask_fraction,
    )

    base_batch = adapter.prepare(image, prompt)
    ablated_batch = adapter.prepare(ablated_image, prompt)

    with torch.no_grad():
        base_outputs = _forward_adapter(adapter, base_batch.model_inputs, requires_grad=False)
        ablated_outputs = _forward_adapter(adapter, ablated_batch.model_inputs, requires_grad=False)

    baseline_score = _score_from_outputs(base_outputs, base_batch.model_inputs)
    ablated_score = _score_from_outputs(ablated_outputs, ablated_batch.model_inputs)
    score_drop = baseline_score - ablated_score

    base_attn = _extract_attention_stack(adapter, base_outputs)
    ablated_attn = _extract_attention_stack(adapter, ablated_outputs)
    layer_count = min(len(base_attn), len(ablated_attn))

    if layer_count == 0:
        raise RuntimeError("No attention layers available for grounding-head discovery.")

    visual_mask = _build_visual_key_mask(
        adapter=adapter,
        batch=base_batch,
        outputs=base_outputs,
        key_len=int(base_attn[0].shape[-1]),
    )

    results: list[GroundingHeadScore] = []
    for layer_idx in range(layer_count):
        attn_a = base_attn[layer_idx]
        attn_b = ablated_attn[layer_idx]

        if attn_a.ndim != 4 or attn_b.ndim != 4:
            continue
        if attn_a.shape != attn_b.shape:
            continue

        mass_a = _head_visual_mass(attn_a, visual_mask)
        mass_b = _head_visual_mass(attn_b, visual_mask)
        delta = mass_a - mass_b

        for head_idx in range(int(delta.shape[0])):
            base_val = float(mass_a[head_idx].item())
            ablate_val = float(mass_b[head_idx].item())
            delta_val = float(delta[head_idx].item())
            grounding = abs(delta_val) * (1.0 + max(score_drop, 0.0))
            results.append(
                GroundingHeadScore(
                    layer_index=layer_idx,
                    head_index=head_idx,
                    baseline_visual_mass=base_val,
                    ablated_visual_mass=ablate_val,
                    delta_visual_mass=delta_val,
                    grounding_score=float(grounding),
                )
            )

    results.sort(key=lambda x: x.grounding_score, reverse=True)
    if top_k > 0:
        results = results[:top_k]

    return GroundingCircuitResult(
        model_family=adapter.family,
        model_name=adapter.model_name,
        prompt=prompt,
        mask_fraction=float(mask_fraction),
        baseline_score=baseline_score,
        ablated_score=ablated_score,
        score_drop=score_drop,
        heads=results,
    )