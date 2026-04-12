"""Attention processing utilities for transformer visualizations."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from multimodallens.utils.tensor_ops import safe_minmax, to_numpy


def stack_attentions(attentions: Sequence[torch.Tensor]) -> torch.Tensor:
    """Stack a tuple/list of attention tensors.

    Each tensor is expected to be [B, H, Q, K].
    Returns shape [L, B, H, Q, K].
    """
    if not attentions:
        raise ValueError("No attention tensors provided.")
    return torch.stack([a for a in attentions], dim=0)


def attention_rollout(attentions: Sequence[torch.Tensor], add_residual: bool = True) -> torch.Tensor:
    """Compute attention rollout across layers.

    Args:
        attentions: Sequence of [B, H, S, S]
        add_residual: add identity before row-normalization.
    Returns:
        Joint attentions [B, S, S]
    """
    attn = stack_attentions(attentions)  # [L, B, H, S, S]
    attn = attn.mean(dim=2)  # [L, B, S, S]

    bsz, seq_len = attn.shape[1], attn.shape[-1]
    joint = torch.eye(seq_len, device=attn.device).unsqueeze(0).repeat(bsz, 1, 1)

    for layer_attn in attn:
        if add_residual:
            layer_attn = layer_attn + torch.eye(seq_len, device=attn.device)
        layer_attn = layer_attn / layer_attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        joint = layer_attn @ joint

    return joint


def last_layer_mean_attention(attentions: Sequence[torch.Tensor]) -> torch.Tensor:
    """Return last-layer mean-head attention [B, S, S]."""
    return attentions[-1].mean(dim=1)


def vector_to_patch_grid(vec: torch.Tensor | np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """Reshape a vector to patch grid with min-max normalization."""
    array = to_numpy(vec).reshape(grid)
    return safe_minmax(array)
