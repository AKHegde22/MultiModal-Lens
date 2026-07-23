"""Multimodal logit lens over intermediate layer activations."""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from multimodallens.adapters.base import ModelAdapter
from multimodallens.analysis.hooking import list_hookable_layers
from multimodallens.core.hooks import ForwardHookCache
from multimodallens.types import LogitLensResult, LogitLensStep
from dataclasses import dataclass

@dataclass
class VisionLogitLensStep:
    layer_name: str
    position: int
    top_tokens: list[str]
    top_probabilities: list[float]
    stage: str
    patch_index: int | None = None

@dataclass 
class VisionLogitLensResult:
    model_family: str
    model_name: str
    stages: dict[str, list[VisionLogitLensStep]]


def _forward_adapter(adapter: ModelAdapter, model_inputs: dict[str, Any], requires_grad: bool = False) -> Any:
    forward_fn = getattr(adapter, "_forward", None)
    if not callable(forward_fn):
        raise RuntimeError(
            f"Adapter '{adapter.__class__.__name__}' does not expose a callable _forward method."
        )
    return forward_fn(model_inputs, requires_grad=requires_grad)


def _get_unembedding(adapter: ModelAdapter) -> torch.nn.Module | None:
    assert adapter.model is not None

    getter = getattr(adapter.model, "get_output_embeddings", None)
    if callable(getter):
        head = getter()
        if head is not None:
            return head

    lm_head = getattr(adapter.model, "lm_head", None)
    if isinstance(lm_head, torch.nn.Module):
        return lm_head

    return None


def _get_embedding_table(adapter: ModelAdapter) -> torch.Tensor | None:
    assert adapter.model is not None

    # Generative models may still expose input embeddings even if output embeddings are absent.
    get_in = getattr(adapter.model, "get_input_embeddings", None)
    if callable(get_in):
        try:
            in_embed = get_in()
        except Exception:
            in_embed = None
        weight = getattr(in_embed, "weight", None)
        if torch.is_tensor(weight):
            return weight

    # CLIP-style models store token embeddings under text tower modules.
    text_model = getattr(adapter.model, "text_model", None)
    text_get_in = getattr(text_model, "get_input_embeddings", None) if text_model is not None else None
    if callable(text_get_in):
        try:
            text_in_embed = text_get_in()
        except Exception:
            text_in_embed = None
        text_weight = getattr(text_in_embed, "weight", None)
        if torch.is_tensor(text_weight):
            return text_weight

    embeddings = getattr(text_model, "embeddings", None) if text_model is not None else None
    token_embedding = getattr(embeddings, "token_embedding", None) if embeddings is not None else None
    weight = getattr(token_embedding, "weight", None)
    if torch.is_tensor(weight):
        return weight

    return None


def _convert_ids_to_tokens(adapter: ModelAdapter, token_ids: list[int]) -> list[str]:
    tokenizer = getattr(adapter, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "convert_ids_to_tokens"):
        try:
            converted = tokenizer.convert_ids_to_tokens(token_ids)
            if isinstance(converted, list):
                return [str(tok) for tok in converted]
        except Exception:
            pass
    return [str(i) for i in token_ids]


def _project_hidden_for_embedding_table(
    adapter: ModelAdapter,
    hidden: torch.Tensor,
    target_dim: int,
) -> torch.Tensor | None:
    if int(hidden.shape[-1]) == int(target_dim):
        return hidden

    if adapter.family != "clip":
        return None

    assert adapter.model is not None
    for proj_name in ("text_projection", "visual_projection"):
        projection = getattr(adapter.model, proj_name, None)
        if projection is None:
            continue

        if callable(projection):
            try:
                projected = projection(hidden)
            except Exception:
                projected = None
            if torch.is_tensor(projected) and int(projected.shape[-1]) == int(target_dim):
                return projected

        if torch.is_tensor(projection) and projection.ndim == 2:
            if int(hidden.shape[-1]) == int(projection.shape[0]):
                projected = hidden @ projection
                if int(projected.shape[-1]) == int(target_dim):
                    return projected
            if int(hidden.shape[-1]) == int(projection.shape[1]):
                projected = hidden @ projection.transpose(0, 1)
                if int(projected.shape[-1]) == int(target_dim):
                    return projected

    return None


def run_multimodal_logit_lens(
    adapter: ModelAdapter,
    image: Image.Image,
    prompt: str,
    layer_names: list[str] | None = None,
    include_patterns: list[str] | None = None,
    positions: list[int] | None = None,
    top_k: int = 5,
    max_layers: int | None = None,
) -> LogitLensResult:
    """Decode intermediate layer states into vocabulary distributions."""
    adapter.ensure_loaded()
    assert adapter.model is not None

    unembed = _get_unembedding(adapter)
    embed_table = _get_embedding_table(adapter)
    if unembed is None and embed_table is None:
        raise RuntimeError(
            "This model does not expose output or token embeddings for logit-lens decoding."
        )

    batch = adapter.prepare(image, prompt)

    selected = layer_names or list_hookable_layers(adapter, include_patterns=include_patterns)
    if not selected:
        raise RuntimeError("No hookable layers were discovered for logit-lens analysis.")

    if max_layers is not None and max_layers > 0:
        selected = selected[:max_layers]

    pos_list = positions or [-1]
    top_k = max(1, top_k)

    with torch.no_grad():
        with ForwardHookCache(adapter.model, selected, max_tokens=None) as cache:
            _forward_adapter(adapter, batch.model_inputs, requires_grad=False)

    steps: list[LogitLensStep] = []
    for layer_name in selected:
        activation = cache.activations.get(layer_name)
        if activation is None:
            continue

        if activation.ndim == 3:
            hidden_seq = activation[0]
        elif activation.ndim == 2:
            hidden_seq = activation
        else:
            continue

        seq_len = int(hidden_seq.shape[0])
        for raw_pos in pos_list:
            pos = raw_pos if raw_pos >= 0 else seq_len + raw_pos
            if pos < 0 or pos >= seq_len:
                continue

            hidden = hidden_seq[pos].to(device=adapter.device, dtype=torch.float32)
            probs: torch.Tensor
            if unembed is not None:
                try:
                    logits = unembed(hidden)
                except Exception:
                    continue
                probs = torch.softmax(logits.float(), dim=-1)
            else:
                assert embed_table is not None
                table = embed_table.to(device=hidden.device, dtype=hidden.dtype)
                decode_hidden = _project_hidden_for_embedding_table(adapter, hidden, int(table.shape[-1]))
                if decode_hidden is None:
                    continue
                token_scores = decode_hidden @ table.transpose(0, 1)
                probs = torch.softmax(token_scores.float(), dim=-1)

            k = min(top_k, int(probs.shape[-1]))
            values, indices = probs.topk(k)

            idx_list = [int(i) for i in indices.detach().cpu().tolist()]
            token_list = _convert_ids_to_tokens(adapter, idx_list)
            prob_list = [float(v) for v in values.detach().cpu().tolist()]

            steps.append(
                LogitLensStep(
                    layer_name=layer_name,
                    position=int(pos),
                    top_tokens=token_list,
                    top_probabilities=prob_list,
                )
            )

    if not steps:
        raise RuntimeError(
            "No decodable hidden states found. Try restricting layers with include_patterns."
        )

    return LogitLensResult(
        model_family=adapter.family,
        model_name=adapter.model_name,
        prompt=prompt,
        steps=steps,
    )


def run_vision_logit_lens(
    adapter: ModelAdapter,
    image: Image.Image,
    layer_names: list[str] | None = None,
    top_k: int = 5,
) -> VisionLogitLensResult:
    """Implement 3-stage vision logit lens."""
    adapter.ensure_loaded()
    assert adapter.model is not None

    unembed = _get_unembedding(adapter)
    embed_table = _get_embedding_table(adapter)
    if unembed is None and embed_table is None:
        raise RuntimeError("No embeddings found for decoding.")

    all_layers = list_hookable_layers(adapter)
    
    vision_patterns = [r"vision", r"visual", r"encoder"]
    pre_proj_layers = []
    for l in all_layers:
        if any(p in l.lower() for p in vision_patterns):
            pre_proj_layers.append(l)
            
    proj_patterns = ['projector', 'connector', 'merger', 'visual_projection', 'multi_modal_projector']
    post_proj_layers = []
    for l in all_layers:
        if any(p in l.lower() for p in proj_patterns):
            post_proj_layers.append(l)
            
    lang_patterns = [r"model.layers", r"language_model", r"decoder", r"transformer.h", r"lm_head"]
    lang_layers = []
    for l in all_layers:
        if any(p in l.lower() for p in lang_patterns):
            lang_layers.append(l)

    if layer_names:
        pre_proj_layers = [l for l in pre_proj_layers if l in layer_names]
        post_proj_layers = [l for l in post_proj_layers if l in layer_names]
        lang_layers = [l for l in lang_layers if l in layer_names]

    target_layers = pre_proj_layers + post_proj_layers + lang_layers
    if not target_layers:
        raise RuntimeError("No hookable layers found.")

    batch = adapter.prepare(image, "[VISION_LOGIT_LENS]")
    
    input_ids = batch.model_inputs.get(adapter.config.text_input_key)
    image_token_id = adapter.config.image_token_id
    tokenizer = getattr(adapter, "tokenizer", None)
    if image_token_id is None and tokenizer is not None:
        try:
            image_token_id = tokenizer.convert_tokens_to_ids(adapter.config.image_token_str)
        except Exception:
            image_token_id = None
            
    image_positions = []
    if input_ids is not None and image_token_id is not None:
        token_seq = input_ids[0]
        image_mask = token_seq == image_token_id
        image_positions = image_mask.nonzero(as_tuple=True)[0].cpu().tolist()

    top_k = max(1, top_k)
    
    with torch.no_grad():
        with ForwardHookCache(adapter.model, target_layers, max_tokens=None) as cache:
            _forward_adapter(adapter, batch.model_inputs, requires_grad=False)

    stages: dict[str, list[VisionLogitLensStep]] = {"pre_projector": [], "post_projector": [], "language_layers": []}
    
    def decode_hidden(hidden_states: torch.Tensor, layer_name: str, stage: str, positions_to_use: list[int] | None = None):
        if hidden_states.ndim == 3:
            hidden_seq = hidden_states[0]
        elif hidden_states.ndim == 2:
            hidden_seq = hidden_states
        else:
            return

        seq_len = int(hidden_seq.shape[0])
        pos_list = positions_to_use if positions_to_use is not None else list(range(seq_len))
        
        for pos in pos_list:
            if pos < 0 or pos >= seq_len:
                continue

            hidden = hidden_seq[pos].to(device=adapter.device, dtype=torch.float32)
            probs: torch.Tensor
            if unembed is not None:
                try:
                    logits = unembed(hidden)
                except Exception:
                    continue
                probs = torch.softmax(logits.float(), dim=-1)
            else:
                table = embed_table.to(device=hidden.device, dtype=hidden.dtype)
                decode_hid = _project_hidden_for_embedding_table(adapter, hidden, int(table.shape[-1]))
                if decode_hid is None:
                    continue
                token_scores = decode_hid @ table.transpose(0, 1)
                probs = torch.softmax(token_scores.float(), dim=-1)

            k = min(top_k, int(probs.shape[-1]))
            values, indices = probs.topk(k)

            idx_list = [int(i) for i in indices.detach().cpu().tolist()]
            token_list = _convert_ids_to_tokens(adapter, idx_list)
            prob_list = [float(v) for v in values.detach().cpu().tolist()]

            stages[stage].append(
                VisionLogitLensStep(
                    layer_name=layer_name,
                    position=int(pos),
                    top_tokens=token_list,
                    top_probabilities=prob_list,
                    stage=stage,
                    patch_index=int(pos) if positions_to_use is None else None
                )
            )

    for layer_name in pre_proj_layers:
        activation = cache.activations.get(layer_name)
        if activation is not None:
            decode_hidden(activation, layer_name, "pre_projector")
            
    for layer_name in post_proj_layers:
        activation = cache.activations.get(layer_name)
        if activation is not None:
            decode_hidden(activation, layer_name, "post_projector")
            
    for layer_name in lang_layers:
        activation = cache.activations.get(layer_name)
        if activation is not None and image_positions:
            decode_hidden(activation, layer_name, "language_layers", positions_to_use=image_positions)

    return VisionLogitLensResult(
        model_family=adapter.family,
        model_name=adapter.model_name,
        stages=stages
    )