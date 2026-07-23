"""HookedVLM: TransformerLens-style main interface for vision-language models."""

from __future__ import annotations

from typing import Any, Callable
import torch
from PIL import Image

from multimodallens.adapters.base import ModelAdapter
from multimodallens.analysis.hooking import list_hookable_layers
from multimodallens.core.activation_cache import ActivationCache
from multimodallens.core.hooks import ForwardHookCache, ForwardLayerPatcher
from multimodallens.core.registry import create_adapter
from multimodallens.analysis.dla import DLAResult, HeadContribution, MLPContribution, run_multimodal_dla
from multimodallens.analysis.logit_lens import LogitLensResult, VisionLogitLensResult, run_multimodal_logit_lens, run_vision_logit_lens
from multimodallens.analysis.path_patching import PathPatchingResult, EdgeEffect, run_causal_path_patching
from multimodallens.types import AnalysisResult


class HookedVLM:
    """TransformerLens-style main interface for multimodal model interpretability.

    Wraps model adapters with clean activation caching, intervention hooks,
    and high-level exploratory analysis methods.
    """

    def __init__(self, adapter: ModelAdapter) -> None:
        self._adapter = adapter

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        family: str = "auto",
        device: str = "auto",
        dtype: str = "float16",
        trust_remote_code: bool = False,
        low_cpu_mem_usage: bool = True,
    ) -> HookedVLM:
        """Load a multimodal model wrapped with HookedVLM capabilities."""
        adapter = create_adapter(
            family=family,
            model_name=model_name,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
        adapter.ensure_loaded()
        return cls(adapter)

    @property
    def adapter(self) -> ModelAdapter:
        """Return the underlying model adapter."""
        return self._adapter

    @property
    def model(self) -> Any:
        """Return the underlying PyTorch model module."""
        if self._adapter.model is None:
            self._adapter.ensure_loaded()
        return self._adapter.model

    @property
    def model_name(self) -> str:
        """Return model checkpoint name."""
        return self._adapter.model_name

    @property
    def family(self) -> str:
        """Return model canonical family name."""
        return self._adapter.family

    @property
    def device(self) -> torch.device:
        """Return model runtime device."""
        return self._adapter.device

    def list_hook_points(self, include_patterns: list[str] | None = None) -> list[str]:
        """List all hookable transformer layer names in the model."""
        return list_hookable_layers(self._adapter, include_patterns=include_patterns)

    def analyze(
        self,
        image: Image.Image,
        prompt: str,
        compute_gradients: bool = False,
    ) -> AnalysisResult:
        """Run standard model analysis and return normalized result."""
        return self._adapter.analyze(
            image=image,
            prompt=prompt,
            compute_gradients=compute_gradients,
        )

    def run_with_cache(
        self,
        image: Image.Image,
        prompt: str,
        layer_names: list[str] | None = None,
        include_patterns: list[str] | None = None,
        max_tokens: int | None = 256,
        compute_gradients: bool = False,
    ) -> tuple[AnalysisResult, ActivationCache]:
        """Run forward pass and capture named activations into an ActivationCache."""
        if self._adapter.model is None:
            self._adapter.ensure_loaded()

        if layer_names is None:
            layer_names = list_hookable_layers(self._adapter, include_patterns=include_patterns)

        hook_cache = ForwardHookCache(
            model=self._adapter.model,
            layer_names=layer_names,
            max_tokens=max_tokens,
        )

        with hook_cache:
            result = self._adapter.analyze(
                image=image,
                prompt=prompt,
                compute_gradients=compute_gradients,
            )

        cache = ActivationCache(hook_cache.activations)
        return result, cache

    def run_with_hooks(
        self,
        image: Image.Image,
        prompt: str,
        fwd_hooks: list[tuple[str, Callable[[torch.Tensor], torch.Tensor]]] | None = None,
        compute_gradients: bool = False,
    ) -> AnalysisResult:
        """Run forward pass with arbitrary intervention hook functions."""
        if self._adapter.model is None:
            self._adapter.ensure_loaded()

        if not fwd_hooks:
            return self.analyze(image=image, prompt=prompt, compute_gradients=compute_gradients)

        patchers: list[ForwardLayerPatcher] = []
        for layer_name, hook_fn in fwd_hooks:
            patcher = ForwardLayerPatcher(
                model=self._adapter.model,
                layer_name=layer_name,
                patch_fn=hook_fn,
            )
            patchers.append(patcher)

        try:
            for patcher in patchers:
                patcher.install()
            return self._adapter.analyze(
                image=image,
                prompt=prompt,
                compute_gradients=compute_gradients,
            )
        finally:
            for patcher in patchers:
                patcher.close()

    def dla(
        self,
        image: Image.Image,
        prompt: str,
        target_token: str | int,
        layer_names: list[str] | None = None,
        include_patterns: list[str] | None = None,
    ) -> DLAResult:
        """Run Direct Logit Attribution (DLA) for a target token."""
        return run_multimodal_dla(
            adapter=self._adapter,
            image=image,
            prompt=prompt,
            target_token=target_token,
            layer_names=layer_names,
            include_patterns=include_patterns,
        )

    def path_patch(
        self,
        clean_image: Image.Image,
        clean_prompt: str,
        corrupted_image: Image.Image,
        corrupted_prompt: str,
        sender_layers: list[str] | None = None,
        receiver_layers: list[str] | None = None,
        receiver_channels: list[str] | None = None,
    ) -> PathPatchingResult:
        """Run Causal Path Patching from sender components to receiver components."""
        return run_causal_path_patching(
            adapter=self._adapter,
            clean_image=clean_image,
            clean_prompt=clean_prompt,
            corrupted_image=corrupted_image,
            corrupted_prompt=corrupted_prompt,
            sender_layers=sender_layers,
            receiver_layers=receiver_layers,
        )

    def logit_lens(
        self,
        image: Image.Image,
        prompt: str,
        layer_names: list[str] | None = None,
        include_patterns: list[str] | None = None,
        top_k: int = 5,
    ) -> LogitLensResult:
        """Run multimodal Logit Lens decoding."""
        return run_multimodal_logit_lens(
            adapter=self._adapter,
            image=image,
            prompt=prompt,
            layer_names=layer_names,
            include_patterns=include_patterns,
            top_k=top_k,
        )

    def vision_logit_lens(
        self,
        image: Image.Image,
        layer_names: list[str] | None = None,
        top_k: int = 5,
    ) -> VisionLogitLensResult:
        """Decode vision tower patch activations into language vocabulary."""
        return run_vision_logit_lens(
            adapter=self._adapter,
            image=image,
            layer_names=layer_names,
            top_k=top_k,
        )

    def fold_layer_norms(self) -> None:
        """Fold LayerNorm parameters into adjacent weights for exact decomposition."""
        from multimodallens.core.weight_processing import fold_layer_norms
        fold_layer_norms(self.model)

    def center_writing_weights(self) -> None:
        """Center residual-writing weights for cleaner DLA."""
        from multimodallens.core.weight_processing import center_writing_weights
        center_writing_weights(self.model)

    def center_unembed(self) -> None:
        """Center unembedding matrix for cleaner logit attribution."""
        from multimodallens.core.weight_processing import center_unembed
        center_unembed(self.model)

    def ov_circuit(self, layer: int, head: int) -> 'FactoredMatrix':
        """Get the OV circuit for a specific attention head."""
        from multimodallens.core.factored_matrix import FactoredMatrix
        # Find W_V and W_O for this layer/head
        return FactoredMatrix(None, None)  # Placeholder

    def qk_circuit(self, layer: int, head: int) -> 'FactoredMatrix':
        """Get the QK circuit for a specific attention head."""
        from multimodallens.core.factored_matrix import FactoredMatrix
        # Find W_Q and W_K for this layer/head
        return FactoredMatrix(None, None)  # Placeholder

