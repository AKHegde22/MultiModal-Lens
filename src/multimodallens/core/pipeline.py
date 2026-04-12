"""Top-level orchestration for MultimodalLens analysis."""

from __future__ import annotations

from dataclasses import replace

from PIL import Image

from multimodallens.analysis.activation_patching import run_cross_modal_activation_patch
from multimodallens.analysis.circuits import discover_grounding_heads
from multimodallens.analysis.faithfulness import build_faithfulness_metrics
from multimodallens.analysis.hooking import capture_layer_activations, list_hookable_layers
from multimodallens.analysis.logit_lens import run_multimodal_logit_lens
from multimodallens.core.preflight import ModelPreflightReport, run_model_preflight
from multimodallens.core.registry import create_adapter
from multimodallens.types import (
    ActivationPatchResult,
    AnalysisResult,
    GroundingCircuitResult,
    LayerActivationRun,
    LogitLensResult,
)


class LensPipeline:
    """Stateful adapter manager + analysis entry points."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str, str, str], object] = {}

    def get_adapter(
        self,
        family: str,
        model_name: str,
        device: str = "auto",
        dtype: str = "float16",
        trust_remote_code: bool = False,
        low_cpu_mem_usage: bool = True,
    ):
        """Get cached adapter or construct a new one."""
        key = (family, model_name, device, dtype)
        if key not in self._cache:
            self._cache[key] = create_adapter(
                family=family,
                model_name=model_name,
                device=device,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=low_cpu_mem_usage,
            )
        return self._cache[key]

    def preflight(
        self,
        family: str,
        model_name: str,
        trust_remote_code: bool = False,
    ) -> ModelPreflightReport:
        """Run lightweight compatibility checks before heavy model execution."""
        return run_model_preflight(
            family=family,
            model_name=model_name,
            trust_remote_code=trust_remote_code,
        )

    def analyze(
        self,
        family: str,
        model_name: str,
        image: Image.Image,
        prompt: str,
        device: str = "auto",
        dtype: str = "float16",
        trust_remote_code: bool = False,
        compute_gradients: bool = False,
        run_faithfulness: bool = False,
    ) -> AnalysisResult:
        """Run analysis end-to-end."""
        adapter = self.get_adapter(
            family=family,
            model_name=model_name,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

        result = adapter.analyze(
            image=image,
            prompt=prompt,
            compute_gradients=compute_gradients,
        )

        if run_faithfulness:
            if "vision_rollout" not in result.attention_maps:
                raise RuntimeError("Missing vision rollout map required for faithfulness tests.")

            patch_scores = result.attention_maps["vision_rollout"].reshape(-1)
            grad_map = result.attention_maps.get("vision_grad")

            metrics = build_faithfulness_metrics(
                adapter=adapter,
                image=image,
                prompt=prompt,
                patch_scores=patch_scores,
                patch_grid=result.patch_grid,
                grad_map=grad_map,
            )
            result = replace(result, faithfulness=metrics)

        return result

    def compare_prompts(
        self,
        family: str,
        model_name: str,
        image: Image.Image,
        prompt_a: str,
        prompt_b: str,
        device: str = "auto",
        dtype: str = "float16",
        trust_remote_code: bool = False,
    ) -> tuple[AnalysisResult, AnalysisResult]:
        """Analyze two prompts with same model for side-by-side comparison."""
        res_a = self.analyze(
            family=family,
            model_name=model_name,
            image=image,
            prompt=prompt_a,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        res_b = self.analyze(
            family=family,
            model_name=model_name,
            image=image,
            prompt=prompt_b,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        return res_a, res_b

    def list_layers(
        self,
        family: str,
        model_name: str,
        device: str = "auto",
        dtype: str = "float16",
        trust_remote_code: bool = False,
        include_patterns: list[str] | None = None,
    ) -> list[str]:
        """List hookable transformer block layers for a selected model."""
        adapter = self.get_adapter(
            family=family,
            model_name=model_name,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        return list_hookable_layers(adapter, include_patterns=include_patterns)

    def capture_internals(
        self,
        family: str,
        model_name: str,
        image: Image.Image,
        prompt: str,
        device: str = "auto",
        dtype: str = "float16",
        trust_remote_code: bool = False,
        layer_names: list[str] | None = None,
        include_patterns: list[str] | None = None,
        max_tokens: int | None = 256,
    ) -> LayerActivationRun:
        """Capture hidden vectors at every selected transformer layer."""
        adapter = self.get_adapter(
            family=family,
            model_name=model_name,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        return capture_layer_activations(
            adapter=adapter,
            image=image,
            prompt=prompt,
            layer_names=layer_names,
            include_patterns=include_patterns,
            max_tokens=max_tokens,
        )

    def activation_patch(
        self,
        family: str,
        model_name: str,
        source_image: Image.Image,
        target_image: Image.Image,
        prompt: str,
        device: str = "auto",
        dtype: str = "float16",
        trust_remote_code: bool = False,
        layer_name: str | None = None,
        layer_index: int = 12,
        include_patterns: list[str] | None = None,
        visual_only: bool = True,
    ) -> ActivationPatchResult:
        """Patch source activations into target forward pass for causal tracing."""
        adapter = self.get_adapter(
            family=family,
            model_name=model_name,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        return run_cross_modal_activation_patch(
            adapter=adapter,
            source_image=source_image,
            target_image=target_image,
            prompt=prompt,
            layer_name=layer_name,
            layer_index=layer_index,
            include_patterns=include_patterns,
            visual_only=visual_only,
        )

    def logit_lens(
        self,
        family: str,
        model_name: str,
        image: Image.Image,
        prompt: str,
        device: str = "auto",
        dtype: str = "float16",
        trust_remote_code: bool = False,
        layer_names: list[str] | None = None,
        include_patterns: list[str] | None = None,
        positions: list[int] | None = None,
        top_k: int = 5,
        max_layers: int | None = None,
    ) -> LogitLensResult:
        """Decode intermediate hidden states into token predictions per layer."""
        adapter = self.get_adapter(
            family=family,
            model_name=model_name,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        return run_multimodal_logit_lens(
            adapter=adapter,
            image=image,
            prompt=prompt,
            layer_names=layer_names,
            include_patterns=include_patterns,
            positions=positions,
            top_k=top_k,
            max_layers=max_layers,
        )

    def grounding_heads(
        self,
        family: str,
        model_name: str,
        image: Image.Image,
        prompt: str,
        device: str = "auto",
        dtype: str = "float16",
        trust_remote_code: bool = False,
        mask_fraction: float = 0.3,
        top_k: int = 20,
    ) -> GroundingCircuitResult:
        """Discover heads with strongest visual grounding sensitivity."""
        adapter = self.get_adapter(
            family=family,
            model_name=model_name,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        return discover_grounding_heads(
            adapter=adapter,
            image=image,
            prompt=prompt,
            mask_fraction=mask_fraction,
            top_k=top_k,
        )
