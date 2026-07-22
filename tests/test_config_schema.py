from multimodallens.core.config_schema import get_multimodal_config, MultimodalConfig
from multimodallens.core.registry import resolve_family, create_adapter
from multimodallens.adapters.generic_adapter import GenericVLMAdapter


def test_multimodal_config_resolution():
    cfg_qwen = get_multimodal_config("qwen2_vl")
    assert cfg_qwen.family == "qwen2_vl"
    assert cfg_qwen.image_token_str == "<|image_pad|>"

    cfg_pixtral = get_multimodal_config("pixtral")
    assert cfg_pixtral.family == "pixtral"
    assert cfg_pixtral.image_token_str == "[IMG]"

    fam = resolve_family("qwen2_vl", "Qwen/Qwen2-VL-7B-Instruct")
    assert fam == "qwen2_vl"


def test_generic_adapter_instantiation():
    adapter = create_adapter("qwen2_vl", "Qwen/Qwen2-VL-7B-Instruct")
    assert isinstance(adapter, GenericVLMAdapter)
    assert adapter.family == "qwen2_vl"
