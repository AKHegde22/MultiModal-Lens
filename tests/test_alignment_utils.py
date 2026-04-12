from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from multimodallens.analysis.alignment import cosine_similarity_matrix, token_contribution_scores  # noqa: E402


def test_cosine_similarity_matrix_shape():
    a = torch.randn(4, 8)
    b = torch.randn(9, 8)
    sim = cosine_similarity_matrix(a, b)
    assert sim.shape == (4, 9)


def test_token_contribution_scores():
    mat = np.array([[0.1, 0.4], [0.9, 0.2]], dtype=np.float32)
    scores = token_contribution_scores(mat)
    assert np.allclose(scores, np.array([0.4, 0.9], dtype=np.float32))
