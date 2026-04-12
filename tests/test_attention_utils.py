from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from multimodallens.analysis.attention import attention_rollout  # noqa: E402


def test_attention_rollout_shape():
    attn = [
        torch.rand(1, 2, 5, 5),
        torch.rand(1, 2, 5, 5),
    ]
    out = attention_rollout(attn)
    assert out.shape == (1, 5, 5)
