from __future__ import annotations

from types import SimpleNamespace

from multimodallens.adapters.blip2_adapter import BLIP2Adapter
from multimodallens.adapters.clip_adapter import CLIPAdapter
from multimodallens.adapters.llava_adapter import LlavaAdapter
from multimodallens.core import registry


def test_alias_resolves_to_clip_adapter() -> None:
    adapter = registry.create_adapter(
        family="siglip",
        model_name="dummy/siglip",
        device="cpu",
        dtype="float32",
    )
    assert isinstance(adapter, CLIPAdapter)


def test_alias_resolves_to_blip2_adapter() -> None:
    adapter = registry.create_adapter(
        family="instructblip",
        model_name="dummy/instructblip",
        device="cpu",
        dtype="float32",
    )
    assert isinstance(adapter, BLIP2Adapter)


def test_alias_resolves_to_llava_adapter() -> None:
    adapter = registry.create_adapter(
        family="qwen2_vl",
        model_name="dummy/qwen2-vl",
        device="cpu",
        dtype="float32",
    )
    assert isinstance(adapter, LlavaAdapter)


def test_auto_family_uses_model_type_inference(monkeypatch) -> None:
    monkeypatch.setattr(
        registry.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(model_type="qwen2_vl", architectures=[]),
    )

    adapter = registry.create_adapter(
        family="auto",
        model_name="dummy/auto",
        device="cpu",
        dtype="float32",
    )
    assert isinstance(adapter, LlavaAdapter)


def test_unknown_family_uses_config_inference(monkeypatch) -> None:
    monkeypatch.setattr(
        registry.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(model_type="siglip", architectures=[]),
    )

    adapter = registry.create_adapter(
        family="future_multimodal_family",
        model_name="dummy/new-family",
        device="cpu",
        dtype="float32",
    )
    assert isinstance(adapter, CLIPAdapter)


def test_supported_families_include_auto_and_qwen() -> None:
    assert "auto" in registry.SUPPORTED_FAMILIES
    assert "qwen2_vl" in registry.SUPPORTED_FAMILIES
