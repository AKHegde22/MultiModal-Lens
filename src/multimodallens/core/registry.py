"""Adapter registry and factory helpers."""

from __future__ import annotations

import re

from multimodallens.adapters.base import ModelAdapter
from multimodallens.adapters.blip2_adapter import BLIP2Adapter
from multimodallens.adapters.clip_adapter import CLIPAdapter
from multimodallens.adapters.llava_adapter import LlavaAdapter
from transformers import AutoConfig


from multimodallens.exceptions import UnsupportedFamilyError


from multimodallens.adapters.generic_adapter import GenericVLMAdapter
from multimodallens.core.config_schema import BUILTIN_CONFIGS


CANONICAL_ADAPTERS: dict[str, type[ModelAdapter]] = {
    "clip": CLIPAdapter,
    "blip2": BLIP2Adapter,
    "llava": LlavaAdapter,
}


def _normalize_family_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


FAMILY_ALIASES: dict[str, str] = {
    "clip": "clip",
    "siglip": "clip",
    "siglip2": "clip",
    "altclip": "clip",
    "xclip": "clip",
    "chinese_clip": "clip",
    "blip2": "blip2",
    "blip_2": "blip2",
    "instructblip": "blip2",
    "instruct_blip": "blip2",
    "llava": "llava",
    "llava_next": "llava",
    "llava_onevision": "llava",
    "llava_next_video": "llava",
    "qwen2_vl": "qwen2_vl",
    "qwen2_5_vl": "qwen2_vl",
    "pixtral": "pixtral",
    "idefics2": "idefics3",
    "idefics3": "idefics3",
    "paligemma": "paligemma",
    "mllama": "llava",
    "internvl": "llava",
    "minicpmv": "llava",
    "minicpmo": "llava",
    "smolvlm": "llava",
    "kosmos2": "llava",
    "florence2": "llava",
}


MODEL_TYPE_TO_CANONICAL: dict[str, str] = {
    **FAMILY_ALIASES,
    "clip_vision_model": "clip",
}


SUPPORTED_FAMILIES: list[str] = [
    "auto",
    "clip",
    "siglip",
    "siglip2",
    "altclip",
    "xclip",
    "blip2",
    "instructblip",
    "llava",
    "llava_next",
    "llava_onevision",
    "qwen2_vl",
    "qwen2_5_vl",
    "pixtral",
    "idefics2",
    "idefics3",
    "paligemma",
    "mllama",
    "internvl",
    "minicpmv",
    "smolvlm",
    "kosmos2",
    "florence2",
]


def infer_family_from_model(model_name: str, trust_remote_code: bool = False) -> str | None:
    """Infer canonical family from Hugging Face model config."""
    try:
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    except Exception:
        return None

    model_type = _normalize_family_name(str(getattr(cfg, "model_type", "")))
    mapped = MODEL_TYPE_TO_CANONICAL.get(model_type)
    if mapped is not None:
        return mapped

    arches = [str(x).lower() for x in getattr(cfg, "architectures", [])]
    arch_text = " ".join(arches)
    if "clip" in arch_text and "blip" not in arch_text:
        return "clip"
    if "blip2" in arch_text or "instructblip" in arch_text:
        return "blip2"

    llava_hints = [
        "llava",
        "idefics",
        "paligemma",
        "mllama",
        "qwen2vl",
        "internvl",
        "minicpm",
        "vision2seq",
        "imagetexttotext",
    ]
    if any(hint in arch_text for hint in llava_hints):
        return "llava"

    return None


def resolve_family(family: str, model_name: str, trust_remote_code: bool = False) -> str:
    """Resolve user family label to canonical adapter family."""
    key = _normalize_family_name(family)
    supported_str = ", ".join(SUPPORTED_FAMILIES)

    if key == "auto":
        inferred = infer_family_from_model(model_name=model_name, trust_remote_code=trust_remote_code)
        if inferred:
            return inferred
        raise UnsupportedFamilyError(
            f"Could not automatically infer model family for '{model_name}'. "
            f"Supported families: {supported_str}"
        )

    mapped = FAMILY_ALIASES.get(key)
    if mapped is not None:
        return mapped

    inferred = infer_family_from_model(model_name=model_name, trust_remote_code=trust_remote_code)
    if inferred:
        return inferred

    if key in BUILTIN_CONFIGS:
        return key

    raise UnsupportedFamilyError(f"Unsupported model family '{family}'. Supported families: {supported_str}")


def create_adapter(
    family: str,
    model_name: str,
    device: str = "auto",
    dtype: str = "float16",
    trust_remote_code: bool = False,
    low_cpu_mem_usage: bool = True,
) -> ModelAdapter:
    """Instantiate model adapter for a family."""
    key = resolve_family(
        family=family,
        model_name=model_name,
        trust_remote_code=trust_remote_code,
    )
    cls = CANONICAL_ADAPTERS.get(key)
    if cls is not None:
        return cls(
            model_name=model_name,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )

    return GenericVLMAdapter(
        family=key,
        model_name=model_name,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )

