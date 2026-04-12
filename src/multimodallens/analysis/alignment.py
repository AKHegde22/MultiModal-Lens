"""Cross-modal alignment utilities."""

from __future__ import annotations

import numpy as np
import torch

from multimodallens.utils.tensor_ops import l2_normalize, to_numpy


def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """Compute token-to-patch cosine similarity matrix.

    Args:
        a: Tensor [N, D]
        b: Tensor [M, D]
    Returns:
        Matrix [N, M]
    """
    a_n = l2_normalize(a, dim=-1)
    b_n = l2_normalize(b, dim=-1)
    sim = a_n @ b_n.transpose(-1, -2)
    return to_numpy(sim)


def token_contribution_scores(alignment_matrix: np.ndarray) -> np.ndarray:
    """Token contribution proxy from token-patch similarities.

    We use each token's max similarity over all visual patches.
    """
    if alignment_matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return alignment_matrix.max(axis=1)
