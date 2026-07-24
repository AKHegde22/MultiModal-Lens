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

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 20,
        **kwargs: Any,
    ) -> str:
        """Run autoregressive text generation."""
        return self._adapter.generate(image=image, prompt=prompt, max_new_tokens=max_new_tokens, **kwargs)

    def generate_with_hooks(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 20,
        fwd_hooks: list[tuple[str, Callable[[torch.Tensor], torch.Tensor]]] | None = None,
        **kwargs: Any,
    ) -> str:
        """Run autoregressive text generation with forward intervention hooks installed."""
        if self._adapter.model is None:
            self._adapter.ensure_loaded()

        if not fwd_hooks:
            return self.generate(image=image, prompt=prompt, max_new_tokens=max_new_tokens, **kwargs)

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
            return self._adapter.generate(
                image=image,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                **kwargs,
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

    def _find_attn_submodules(self, layer: int) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.nn.Module]:
        """Find q_proj, k_proj, v_proj, o_proj for a given layer index."""
        model = self.model
        attn_module = None
        for name, module in model.named_modules():
            if (f"layers.{layer}.self_attn" in name or
                f"layer.{layer}.attention" in name or
                f"blocks.{layer}.attn" in name or
                f"layers.{layer}.attn" in name or
                f"h.{layer}.attn" in name):
                attn_module = module
                break

        if attn_module is None:
            for name, module in model.named_modules():
                segments = name.split(".")
                if len(segments) >= 2 and segments[-1] in ("self_attn", "attn", "attention") and (f".{layer}." in name or name.endswith(f".{layer}")):
                    attn_module = module
                    break

        if attn_module is None:
            raise ValueError(f"Could not locate attention module for layer {layer}.")

        q_proj = getattr(attn_module, "q_proj", getattr(attn_module, "q", None))
        k_proj = getattr(attn_module, "k_proj", getattr(attn_module, "k", None))
        v_proj = getattr(attn_module, "v_proj", getattr(attn_module, "v", None))
        o_proj = getattr(attn_module, "o_proj", getattr(attn_module, "out_proj", getattr(attn_module, "c_proj", None)))

        if any(proj is None for proj in (q_proj, k_proj, v_proj, o_proj)):
            sub_linears = {n: m for n, m in attn_module.named_modules() if isinstance(m, torch.nn.Linear)}
            for n, m in sub_linears.items():
                if "q" in n and q_proj is None:
                    q_proj = m
                elif "k" in n and k_proj is None:
                    k_proj = m
                elif "v" in n and v_proj is None:
                    v_proj = m
                elif ("o" in n or "out" in n or "proj" in n) and o_proj is None:
                    o_proj = m

        if any(proj is None for proj in (q_proj, k_proj, v_proj, o_proj)):
            raise RuntimeError(f"Could not resolve Q/K/V/O projection matrices for layer {layer}.")

        return q_proj, k_proj, v_proj, o_proj

    def _extract_head_matrices(self, layer: int, head: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract W_Q, W_K, W_V, W_O for specific head of layer."""
        q_proj, k_proj, v_proj, o_proj = self._find_attn_submodules(layer)

        w_q = q_proj.weight  # [num_heads * d_head, d_model]
        w_k = k_proj.weight  # [num_heads * d_head, d_model]
        w_v = v_proj.weight  # [num_heads * d_head, d_model]
        w_o = o_proj.weight  # [d_model, num_heads * d_head]

        num_heads = getattr(self.model.config, "num_attention_heads", None)
        if num_heads is None:
            num_heads = getattr(self.model.config, "num_heads", getattr(self.model.config, "n_head", None))
        if num_heads is None:
            num_heads = 8

        d_head = w_q.shape[0] // num_heads

        W_Q = w_q.view(num_heads, d_head, -1)[head].T
        W_K = w_k.view(num_heads, d_head, -1)[head].T
        W_V = w_v.view(num_heads, d_head, -1)[head].T
        W_O = w_o.view(-1, num_heads, d_head)[:, head, :].T

        return W_Q, W_K, W_V, W_O

    def ov_circuit(self, layer: int, head: int) -> FactoredMatrix:
        """Get the OV circuit for a specific attention head."""
        from multimodallens.core.factored_matrix import FactoredMatrix
        _, _, W_V, W_O = self._extract_head_matrices(layer, head)
        return FactoredMatrix.ov_circuit(W_V, W_O)

    def qk_circuit(self, layer: int, head: int) -> FactoredMatrix:
        """Get the QK circuit for a specific attention head."""
        from multimodallens.core.factored_matrix import FactoredMatrix
        W_Q, W_K, _, _ = self._extract_head_matrices(layer, head)
        return FactoredMatrix.qk_circuit(W_Q, W_K)


