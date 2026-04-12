from __future__ import annotations

from types import SimpleNamespace

from multimodallens.core import preflight


def test_preflight_resolves_auto_family_with_vision_config(monkeypatch) -> None:
    monkeypatch.setattr(
        preflight.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(
            model_type="qwen2_vl",
            architectures=["Qwen2VLForConditionalGeneration"],
            vision_config=SimpleNamespace(),
            auto_map=None,
        ),
    )

    report = preflight.run_model_preflight(
        family="auto",
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        trust_remote_code=False,
    )

    assert report.resolved_family == "llava"
    assert report.has_vision_tower
    assert report.supports_explore
    assert not report.errors


def test_preflight_warns_for_non_multimodal_config(monkeypatch) -> None:
    monkeypatch.setattr(
        preflight.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(
            model_type="gemma2",
            architectures=["GemmaForCausalLM"],
            auto_map=None,
        ),
    )

    report = preflight.run_model_preflight(
        family="auto",
        model_name="google/gemma-2-2b",
        trust_remote_code=False,
    )

    assert not report.has_vision_tower
    assert not report.supports_explore
    assert any("does not clearly expose vision components" in msg for msg in report.warnings)


def test_preflight_warns_for_auto_map_without_trust(monkeypatch) -> None:
    monkeypatch.setattr(
        preflight.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(
            model_type="idefics2",
            architectures=["Idefics2ForConditionalGeneration"],
            vision_config=SimpleNamespace(),
            auto_map={"AutoModel": "custom.module.CustomModel"},
        ),
    )

    report = preflight.run_model_preflight(
        family="auto",
        model_name="HuggingFaceM4/idefics2-8b",
        trust_remote_code=False,
    )

    assert report.requires_trust_remote_code
    assert any("trust_remote_code=True" in msg for msg in report.warnings)
