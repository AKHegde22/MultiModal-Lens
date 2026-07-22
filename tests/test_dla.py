from multimodallens.analysis.dla import run_multimodal_dla, DLAResult
from tests.test_mechanistic_features import _ToyAdapter, _make_image


def test_multimodal_dla():
    adapter = _ToyAdapter(model_name="toy")
    adapter.load()
    image = _make_image(1)

    res = run_multimodal_dla(adapter, image, "a cat", target_token=0)
    assert isinstance(res, DLAResult)
    assert res.target_token == "0"
    assert len(res.contributions) > 0
    assert hasattr(res, "to_html")
