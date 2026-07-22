from multimodallens.analysis.path_patching import run_causal_path_patching, PathPatchingResult
from tests.test_mechanistic_features import _ToyAdapter, _make_image


def test_causal_path_patching():
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
    )
    assert isinstance(res, PathPatchingResult)
    assert len(res.effects) > 0
    assert "html" in res.to_html().lower()
