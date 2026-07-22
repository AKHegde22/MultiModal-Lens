"""Activation cache data structure for storing layer activations."""

from __future__ import annotations

import fnmatch
from typing import Callable, Iterator
import torch


class ActivationCache:
    """Dictionary-like container for captured transformer activations.

    Provides intuitive access and utilities for accessing and manipulating internal model activations.
    """

    def __init__(self, activations: dict[str, torch.Tensor]) -> None:
        self._cache: dict[str, torch.Tensor] = activations

    def __getitem__(self, key: str) -> torch.Tensor:
        """Get activation tensor by exact layer name or pattern.

        If exact key isn't found, performs glob pattern matching.
        """
        if key in self._cache:
            return self._cache[key]

        # Try glob pattern match (e.g. "*.layers.5.*")
        matches = [k for k in self._cache if fnmatch.fnmatch(k, key)]
        if len(matches) == 1:
            return self._cache[matches[0]]
        if len(matches) > 1:
            raise KeyError(
                f"Pattern '{key}' matched multiple layers in cache: {matches}. "
                "Specify a more exact layer name."
            )
        raise KeyError(f"Layer '{key}' not found in ActivationCache. Available layers: {list(self._cache.keys())}")

    def __contains__(self, key: str) -> bool:
        if key in self._cache:
            return True
        return len([k for k in self._cache if fnmatch.fnmatch(k, key)]) > 0

    def __len__(self) -> int:
        return len(self._cache)

    def __iter__(self) -> Iterator[str]:
        return iter(self._cache)

    def keys(self) -> list[str]:
        """Return list of all cached layer names."""
        return list(self._cache.keys())

    def values(self) -> list[torch.Tensor]:
        """Return list of cached activation tensors."""
        return list(self._cache.values())

    def items(self) -> list[tuple[str, torch.Tensor]]:
        """Return list of (layer_name, tensor) pairs."""
        return list(self._cache.items())

    def has_key(self, key: str) -> bool:
        """Check if layer name or pattern exists in cache."""
        return key in self

    def to(self, device: str | torch.device) -> ActivationCache:
        """Move all cached tensors to specified device."""
        target_device = torch.device(device) if isinstance(device, str) else device
        moved = {k: v.to(target_device) for k, v in self._cache.items()}
        return ActivationCache(moved)

    def apply_to_cache(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> ActivationCache:
        """Apply a transformation function to all cached tensors."""
        transformed = {k: fn(v) for k, v in self._cache.items()}
        return ActivationCache(transformed)

    def __repr__(self) -> str:
        layers_str = ", ".join(f"'{k}': {tuple(v.shape)}" for k, v in self._cache.items())
        return f"ActivationCache({{{layers_str}}})"
