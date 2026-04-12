"""Tensor utility helpers."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

try:  # pragma: no cover - exercised in runtime environments with torch installed
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def to_numpy(tensor: Any | np.ndarray) -> np.ndarray:
    """Detach and convert tensor to numpy."""
    if isinstance(tensor, np.ndarray):
        return tensor
    if torch is None:
        raise RuntimeError("torch is required to convert non-numpy tensors.")
    return tensor.detach().float().cpu().numpy()


def l2_normalize(tensor: Any, dim: int = -1, eps: float = 1e-8) -> Any:
    """L2 normalize along a dimension."""
    if torch is None:
        raise RuntimeError("torch is required for l2_normalize.")
    return tensor / (tensor.norm(p=2, dim=dim, keepdim=True).clamp_min(eps))


def safe_minmax(array: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Scale an array to [0, 1] robustly."""
    a_min = float(array.min())
    a_max = float(array.max())
    denom = max(a_max - a_min, eps)
    return (array - a_min) / denom


def mean_tensors(tensors: Iterable[Any]) -> Any:
    """Mean over a list/tuple of tensors."""
    if torch is None:
        raise RuntimeError("torch is required for mean_tensors.")
    stacked = torch.stack(list(tensors), dim=0)
    return stacked.mean(dim=0)
