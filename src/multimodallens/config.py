"""Runtime configuration defaults."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_MODELS: dict[str, str] = {
    "auto": "openai/clip-vit-base-patch32",
    "clip": "openai/clip-vit-base-patch32",
    "siglip": "google/siglip-base-patch16-224",
    "siglip2": "google/siglip2-base-patch16-224",
    "altclip": "openai/clip-vit-base-patch32",
    "xclip": "openai/clip-vit-base-patch32",
    "blip2": "Salesforce/blip2-opt-2.7b",
    "instructblip": "Salesforce/instructblip-vicuna-7b",
    "llava": "llava-hf/llava-1.5-7b-hf",
    "llava_next": "llava-hf/llava-v1.6-mistral-7b-hf",
    "llava_onevision": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    "qwen2_vl": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen2_5_vl": "Qwen/Qwen2.5-VL-3B-Instruct",
    "idefics2": "HuggingFaceM4/idefics2-8b",
    "idefics3": "HuggingFaceM4/Idefics3-8B-Llama3",
    "paligemma": "google/paligemma-3b-mix-224",
    "mllama": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "internvl": "OpenGVLab/InternVL2_5-2B",
    "minicpmv": "openbmb/MiniCPM-V-2_6",
    "smolvlm": "HuggingFaceTB/SmolVLM-Instruct",
    "kosmos2": "microsoft/kosmos-2-patch14-224",
    "florence2": "microsoft/Florence-2-base",
}


@dataclass(slots=True)
class RuntimeConfig:
    """Config options for loading Hugging Face models."""

    device: str = "auto"
    dtype: str = "float16"
    trust_remote_code: bool = False
    low_cpu_mem_usage: bool = True
