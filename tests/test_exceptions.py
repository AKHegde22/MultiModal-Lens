from __future__ import annotations

import warnings
import pytest

from multimodallens.adapters.base import ModelAdapter
from multimodallens.core.registry import resolve_family
from multimodallens.exceptions import (
    MultimodalLensError,
    UnsupportedDtypeError,
    UnsupportedFamilyError,
)


def test_exception_inheritance():
    err = UnsupportedFamilyError("test")
    assert isinstance(err, MultimodalLensError)


def test_unknown_family_raises_unsupported_family_error():
    with pytest.raises(UnsupportedFamilyError, match="Unsupported model family"):
        resolve_family(family="non_existent_family_12345", model_name="unknown/model")


def test_invalid_dtype_raises_unsupported_dtype_error():
    class DummyAdapter(ModelAdapter):
        family = "clip"
        def load(self): pass
        def prepare(self, image, prompt): pass
        def analyze(self, image, prompt, compute_gradients=False): pass
        def score(self, image, prompt): pass

    with pytest.raises(UnsupportedDtypeError, match="Unsupported dtype"):
        DummyAdapter(model_name="dummy", device="cpu", dtype="invalid_dtype_name")


def test_cpu_float16_emits_warning():
    class DummyAdapter(ModelAdapter):
        family = "clip"
        def load(self): pass
        def prepare(self, image, prompt): pass
        def analyze(self, image, prompt, compute_gradients=False): pass
        def score(self, image, prompt): pass

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        adapter = DummyAdapter(model_name="dummy", device="cpu", dtype="float16")
        assert adapter.dtype_name == "float32"
        assert len(w) == 1
        assert "Promoting precision to float32" in str(w[0].message)
