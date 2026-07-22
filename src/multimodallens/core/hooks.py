"""Forward-hook helpers for layer-wise activation capture and patching."""

from __future__ import annotations

from collections.abc import Iterable
import re
from typing import Any, Callable

import torch


LAYER_INDEX_PARENTS = {"layers", "layer", "h", "blocks", "block"}


KNOWN_SUBMODULE_SUFFIXES = {
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj", "act_fn",
    "self_attn", "mlp", "post_attention_layernorm", "input_layernorm",
    "multi_modal_projector", "projector", "visual_projection", "text_projection",
}


def discover_transformer_layers(
    model: torch.nn.Module,
    include_patterns: Iterable[str] | None = None,
    include_submodules: bool = True,
) -> list[str]:
    """Discover hookable transformer block paths and fine-grained submodules.

    Discovers main layer blocks (e.g. ``encoder.layers.5``) as well as internal
    submodules (e.g. ``encoder.layers.5.self_attn.q_proj``, ``mlp``, ``projector``).
    """
    include_regex = [re.compile(p) for p in include_patterns or []]

    layer_names: list[str] = []
    for name, _module in model.named_modules():
        if not name:
            continue
        segments = name.split(".")
        if len(segments) < 2:
            continue

        is_main_layer = segments[-1].isdigit() and segments[-2] in LAYER_INDEX_PARENTS
        is_submodule = (
            include_submodules
            and len(segments) >= 3
            and (
                segments[-1] in KNOWN_SUBMODULE_SUFFIXES
                or any(seg.isdigit() for seg in segments[:-1])
            )
        )

        if not (is_main_layer or is_submodule):
            continue

        if include_regex and not any(r.search(name) for r in include_regex):
            continue

        layer_names.append(name)
    return layer_names


def _extract_first_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value

    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _extract_first_tensor(item)
            if tensor is not None:
                return tensor
        return None

    if isinstance(value, dict):
        for item in value.values():
            tensor = _extract_first_tensor(item)
            if tensor is not None:
                return tensor
        return None

    last_hidden = getattr(value, "last_hidden_state", None)
    if torch.is_tensor(last_hidden):
        return last_hidden

    return None


def _replace_first_tensor(value: Any, new_tensor: torch.Tensor) -> Any:
    if torch.is_tensor(value):
        return new_tensor

    if isinstance(value, tuple):
        replaced = False
        out: list[Any] = []
        for item in value:
            if not replaced and _extract_first_tensor(item) is not None:
                out.append(_replace_first_tensor(item, new_tensor))
                replaced = True
            else:
                out.append(item)
        return tuple(out)

    if isinstance(value, list):
        replaced = False
        out_list: list[Any] = []
        for item in value:
            if not replaced and _extract_first_tensor(item) is not None:
                out_list.append(_replace_first_tensor(item, new_tensor))
                replaced = True
            else:
                out_list.append(item)
        return out_list

    if isinstance(value, dict):
        replaced = False
        out_dict: dict[str, Any] = {}
        for key, item in value.items():
            if not replaced and _extract_first_tensor(item) is not None:
                out_dict[key] = _replace_first_tensor(item, new_tensor)
                replaced = True
            else:
                out_dict[key] = item
        return out_dict

    return value


def _limit_tokens(tensor: torch.Tensor, max_tokens: int | None) -> torch.Tensor:
    if max_tokens is None or max_tokens <= 0:
        return tensor
    if tensor.ndim == 2 and tensor.shape[0] > max_tokens:
        return tensor[:max_tokens]
    if tensor.ndim >= 3 and tensor.shape[1] > max_tokens:
        return tensor[:, :max_tokens]
    return tensor


class ForwardHookCache:
    """Capture layer outputs using ``register_forward_hook``."""

    def __init__(
        self,
        model: torch.nn.Module,
        layer_names: list[str],
        max_tokens: int | None = None,
    ) -> None:
        if not layer_names:
            raise ValueError("No layer names were provided for hook capture.")

        self.model = model
        self.layer_names = layer_names
        self.max_tokens = max_tokens
        self.activations: dict[str, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _hook(self, layer_name: str):
        def _capture(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = _extract_first_tensor(output)
            if tensor is None:
                return
            clipped = _limit_tokens(tensor, self.max_tokens)
            self.activations[layer_name] = clipped.detach().cpu().clone()

        return _capture

    def install(self) -> None:
        modules = dict(self.model.named_modules())
        for name in self.layer_names:
            module = modules.get(name)
            if module is None:
                raise ValueError(f"Layer '{name}' was not found in model modules.")
            self._handles.append(module.register_forward_hook(self._hook(name)))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "ForwardHookCache":
        self.install()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


class ForwardLayerPatcher:
    """Apply an output patch function at a single layer during forward."""

    def __init__(
        self,
        model: torch.nn.Module,
        layer_name: str,
        patch_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        self.model = model
        self.layer_name = layer_name
        self.patch_fn = patch_fn
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    def install(self) -> None:
        module = dict(self.model.named_modules()).get(self.layer_name)
        if module is None:
            raise ValueError(f"Layer '{self.layer_name}' was not found in model modules.")

        def _patch(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            tensor = _extract_first_tensor(output)
            if tensor is None:
                return output
            patched = self.patch_fn(tensor)
            if patched is tensor:
                return output
            return _replace_first_tensor(output, patched)

        self._handle = module.register_forward_hook(_patch)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> "ForwardLayerPatcher":
        self.install()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()