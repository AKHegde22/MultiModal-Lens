"""HookPoint: First-class hook point container matching TransformerLens semantics."""

from __future__ import annotations

from typing import Callable
import torch
import torch.nn as nn


class HookPoint(nn.Module):
    """Explicit hook point module embedded in computational graphs.

    Passes through input tensors unchanged unless a hook function returns a modified tensor.
    """

    def __init__(self, name: str | None = None) -> None:
        super().__init__()
        self.name = name
        self._hooks: list[Callable[[torch.Tensor], torch.Tensor | None]] = []

    def add_hook(self, hook_fn: Callable[[torch.Tensor], torch.Tensor | None]) -> None:
        """Register an intervention hook callback."""
        self._hooks.append(hook_fn)

    def remove_hooks(self) -> None:
        """Remove all registered hook callbacks."""
        self._hooks.clear()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pass tensor through hook pipeline."""
        if not self._hooks:
            return tensor

        out = tensor
        for hook_fn in self._hooks:
            res = hook_fn(out)
            if res is not None:
                out = res
        return out
