"""Activation cache data structure for storing layer activations."""

from __future__ import annotations

import fnmatch
from typing import Any, Callable, Iterator
import torch


class ActivationCache:
    """Dictionary-like container for captured transformer activations.

    Provides intuitive access and utilities for accessing and manipulating internal model activations.
    """

    def __init__(
        self,
        activations: dict[str, torch.Tensor],
        model_config: dict[str, Any] | None = None,
    ) -> None:
        self._cache: dict[str, torch.Tensor] = activations
        self.model_config = model_config or {}

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
        return ActivationCache(moved, model_config=self.model_config)

    def apply_to_cache(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> ActivationCache:
        """Apply a transformation function to all cached tensors."""
        transformed = {k: fn(v) for k, v in self._cache.items()}
        return ActivationCache(transformed, model_config=self.model_config)

    def filter(self, pattern: str) -> ActivationCache:
        """Return a new ActivationCache containing only keys matching the pattern."""
        filtered = {k: v for k, v in self._cache.items() if fnmatch.fnmatch(k, pattern)}
        return ActivationCache(filtered, model_config=self.model_config)

    def remove(self, pattern: str) -> ActivationCache:
        """Return a new ActivationCache with keys matching pattern removed."""
        remaining = {k: v for k, v in self._cache.items() if not fnmatch.fnmatch(k, pattern)}
        return ActivationCache(remaining, model_config=self.model_config)

    def accumulated_resid(self, layer: int, incl_mid: bool = False) -> torch.Tensor:
        """Sum of all residual stream writes up to layer L.
        
        Returns: x_embed + sum(layer_outputs[0:layer])
        """
        # Find embed
        embed_keys = [k for k in self._cache if "embed" in k]
        if not embed_keys:
            raise ValueError("No embedding layer found in cache.")
        
        accumulated = self._cache[embed_keys[0]].clone()
        
        for i in range(layer):
            layer_keys = [k for k in self._cache if fnmatch.fnmatch(k, f"*.layers.{i}") or fnmatch.fnmatch(k, f"*.blocks.{i}")]
            if layer_keys:
                accumulated += self._cache[layer_keys[0]]
            
        return accumulated

    def decompose_resid(self, layer: int) -> torch.Tensor:
        """Break residual stream into individual components.
        
        Returns a stacked tensor of shape (n_components, batch, seq, d_model)
        where components are: [embed, layer_0, layer_1, ..., layer_{L-1}]
        """
        components = []
        
        embed_keys = [k for k in self._cache if "embed" in k]
        if not embed_keys:
            raise ValueError("No embedding layer found in cache.")
        components.append(self._cache[embed_keys[0]])
        
        for i in range(layer):
            layer_keys = [k for k in self._cache if fnmatch.fnmatch(k, f"*.layers.{i}") or fnmatch.fnmatch(k, f"*.blocks.{i}")]
            if layer_keys:
                components.append(self._cache[layer_keys[0]])
            else:
                raise ValueError(f"Layer {i} output not found in cache.")
                
        return torch.stack(components, dim=0)

    def stack_head_results(self, layer: int | None = None) -> torch.Tensor:
        """Stack per-head outputs if available in cache.
        
        Looks for keys matching *.self_attn or *.attn
        If specific layer is given, returns just that layer's attention output.
        """
        if layer is not None:
            layer_keys = [k for k in self._cache if fnmatch.fnmatch(k, f"*.layers.{layer}.self_attn") or 
                          fnmatch.fnmatch(k, f"*.blocks.{layer}.attn") or
                          fnmatch.fnmatch(k, f"*.layers.{layer}.attn") or
                          fnmatch.fnmatch(k, f"*.blocks.{layer}.self_attn")]
            if not layer_keys:
                raise ValueError(f"Attention output for layer {layer} not found.")
            return self._cache[layer_keys[0]]
        else:
            attn_keys = [k for k in self._cache if fnmatch.fnmatch(k, "*.self_attn") or fnmatch.fnmatch(k, "*.attn")]
            if not attn_keys:
                raise ValueError("No attention outputs found.")
            return torch.stack([self._cache[k] for k in attn_keys], dim=0)

    def __repr__(self) -> str:
        layers_str = ", ".join(f"'{k}': {tuple(v.shape)}" for k, v in self._cache.items())
        return f"ActivationCache({{{layers_str}}})"
