from multimodallens.analysis.path_patching import run_causal_path_patching, PathPatchingResult
from tests.test_mechanistic_features import _ToyAdapter, _make_image


def test_causal_path_patching():
    adapter = _ToyAdapter(model_name="toy")
    adapter.load()
    image1 = _make_image(1)
    image2 = _make_image(2)

    # The toy model doesn't have self_attn.q_proj submodules,
    # so we use receiver_channels=["residual"] for layer-level patching
    res = run_causal_path_patching(
        adapter=adapter,
        clean_image=image1,
        clean_prompt="a cat",
        corrupted_image=image2,
        corrupted_prompt="a dog",
        receiver_channels=["residual"],
    )
    assert isinstance(res, PathPatchingResult)
    assert len(res.effects) > 0
    assert "html" in res.to_html().lower()


def test_path_patching_result_to_dict():
    adapter = _ToyAdapter(model_name="toy")
    adapter.load()
    image1 = _make_image(1)
    image2 = _make_image(2)

    res = run_causal_path_patching(
        adapter=adapter,
        clean_image=image1,
        clean_prompt="a cat",
        corrupted_image=image2,
        corrupted_prompt="a dog",
        receiver_channels=["residual"],
    )
    d = res.to_dict()
    assert "effects" in d
    assert "model_family" in d
    for effect in d["effects"]:
        assert "sender_layer" in effect
        assert "receiver_layer" in effect
        assert "receiver_channel" in effect
        assert "causal_effect" in effect
