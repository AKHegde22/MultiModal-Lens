"""MultimodalConfig schema for declarative architecture definitions across Vision-Language Models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MultimodalConfig:
    """Declarative specification mapping VLM model structure and token conventions.

    Allows new VLM families (Qwen2-VL, Pixtral, Idefics3, PaliGemma, LLaVA, etc.)
    to be added via configuration without custom adapter boilerplate code.
    """

    family: str
    vision_tower_path: str = "vision_tower"
    projector_path: str = "multi_modal_projector"
    language_model_path: str = "language_model"
    unembed_module_path: str = "lm_head"
    embed_tokens_path: str = "model.embed_tokens"

    image_token_str: str = "<image>"
    image_token_id: int | None = None
    vision_patch_tokens: int | None = None
    image_input_key: str = "pixel_values"
    text_input_key: str = "input_ids"

    supports_gradients: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


# Built-in architecture specifications
BUILTIN_CONFIGS: dict[str, MultimodalConfig] = {
    "llava": MultimodalConfig(
        family="llava",
        vision_tower_path="vision_tower",
        projector_path="multi_modal_projector",
        language_model_path="language_model",
        unembed_module_path="language_model.lm_head",
        embed_tokens_path="language_model.model.embed_tokens",
        image_token_str="<image>",
        vision_patch_tokens=576,
    ),
    "qwen2_vl": MultimodalConfig(
        family="qwen2_vl",
        vision_tower_path="visual",
        projector_path="visual.merger",
        language_model_path="model",
        unembed_module_path="lm_head",
        embed_tokens_path="model.embed_tokens",
        image_token_str="<|image_pad|>",
        vision_patch_tokens=256,
    ),
    "pixtral": MultimodalConfig(
        family="pixtral",
        vision_tower_path="vision_tower",
        projector_path="multi_modal_projector",
        language_model_path="language_model",
        unembed_module_path="language_model.lm_head",
        embed_tokens_path="language_model.model.embed_tokens",
        image_token_str="[IMG]",
    ),
    "idefics3": MultimodalConfig(
        family="idefics3",
        vision_tower_path="model.vision_model",
        projector_path="model.connector",
        language_model_path="model.text_model",
        unembed_module_path="lm_head",
        embed_tokens_path="model.text_model.embed_tokens",
        image_token_str="<image>",
    ),
    "paligemma": MultimodalConfig(
        family="paligemma",
        vision_tower_path="vision_tower",
        projector_path="multi_modal_projector",
        language_model_path="language_model",
        unembed_module_path="language_model.lm_head",
        embed_tokens_path="language_model.model.embed_tokens",
        image_token_str="<image>",
        vision_patch_tokens=256,
    ),
    "clip": MultimodalConfig(
        family="clip",
        vision_tower_path="vision_model",
        projector_path="visual_projection",
        language_model_path="text_model",
        unembed_module_path="text_projection",
        embed_tokens_path="text_model.embeddings.token_embedding",
        image_token_str="<image>",
        vision_patch_tokens=50,
    ),
    "blip2": MultimodalConfig(
        family="blip2",
        vision_tower_path="vision_model",
        projector_path="qformer",
        language_model_path="language_model",
        unembed_module_path="language_model.lm_head",
        embed_tokens_path="language_model.model.embed_tokens",
        image_token_str="<image>",
        vision_patch_tokens=32,
    ),
}


def get_multimodal_config(family: str) -> MultimodalConfig:
    """Resolve MultimodalConfig by canonical family name, with generic fallback."""
    canonical = family.lower().strip()
    if canonical in BUILTIN_CONFIGS:
        return BUILTIN_CONFIGS[canonical]

    for key, cfg in BUILTIN_CONFIGS.items():
        if key in canonical or canonical in key:
            return cfg

    # Default generic configuration fallback
    return MultimodalConfig(family=canonical)
