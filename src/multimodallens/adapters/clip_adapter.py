"""Adapter implementation for CLIP-like dual encoders."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel

from multimodallens.adapters.base import ModelAdapter
from multimodallens.analysis.alignment import cosine_similarity_matrix, token_contribution_scores
from multimodallens.analysis.attention import attention_rollout, last_layer_mean_attention, vector_to_patch_grid
from multimodallens.types import AnalysisResult, AdapterBatch
from multimodallens.utils.image_ops import infer_patch_layout
from multimodallens.utils.tensor_ops import l2_normalize, to_numpy


class CLIPAdapter(ModelAdapter):
    """Adapter for CLIP/SigLIP-style dual-encoder models."""

    family = "clip"

    def _configure_mechanistic_outputs(self) -> None:
        """Ensure CLIP exposes hidden states and attention tensors for probing."""
        assert self.model is not None

        config_candidates = [
            getattr(self.model, "config", None),
            getattr(getattr(self.model, "text_model", None), "config", None),
            getattr(getattr(self.model, "vision_model", None), "config", None),
        ]
        for config in config_candidates:
            if config is None:
                continue
            config.output_attentions = True
            config.output_hidden_states = True
            config.return_dict = True
            if hasattr(config, "_attn_implementation"):
                config._attn_implementation = "eager"

    def load(self) -> None:
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        model_kwargs = {
            "torch_dtype": self.torch_dtype if self.device.type != "cpu" else torch.float32,
            "trust_remote_code": self.trust_remote_code,
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
        }

        # Eager attention keeps attention tensors materialized for grounding-head discovery.
        try:
            self.model = CLIPModel.from_pretrained(
                self.model_name,
                attn_implementation="eager",
                **model_kwargs,
            )
        except TypeError:
            self.model = CLIPModel.from_pretrained(
                self.model_name,
                **model_kwargs,
            )

        self._configure_mechanistic_outputs()
        self.model.to(self.device)  # type: ignore[arg-type]
        self.model.eval()

    def prepare(self, image: Image.Image, prompt: str) -> AdapterBatch:
        self.ensure_loaded()
        assert self.processor is not None

        packed = self.processor(text=[prompt], images=image, return_tensors="pt", padding=True)
        token_ids = packed["input_ids"][0].tolist() if "input_ids" in packed else []

        if self.tokenizer is not None and token_ids:
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        else:
            tokens = [str(i) for i in token_ids]

        return AdapterBatch(
            model_inputs=self._move_inputs(dict(packed)),
            tokens=tokens,
            token_ids=token_ids,
        )

    def _forward(self, model_inputs: dict[str, Any], requires_grad: bool) -> Any:
        assert self.model is not None
        if requires_grad:
            self.model.zero_grad(set_to_none=True)
            return self.model(**model_inputs, **self._forward_kwargs())
        with torch.no_grad():
            return self.model(**model_inputs, **self._forward_kwargs())

    def analyze(
        self,
        image: Image.Image,
        prompt: str,
        compute_gradients: bool = False,
    ) -> AnalysisResult:
        self.ensure_loaded()
        assert self.model is not None

        batch = self.prepare(image, prompt)
        outputs = self._forward(batch.model_inputs, requires_grad=compute_gradients)

        vision_out = outputs.vision_model_output
        text_out = outputs.text_model_output

        image_hidden = vision_out.last_hidden_state[0, 1:]
        num_patches = int(image_hidden.shape[0])
        layout = infer_patch_layout(num_patches)
        patch_grid = (layout.rows, layout.cols)

        if getattr(vision_out, "attentions", None):
            rollout = attention_rollout(vision_out.attentions)[0]
            cls_to_patches = rollout[0, 1 : 1 + num_patches]
            vision_rollout_map = vector_to_patch_grid(cls_to_patches, patch_grid)
        else:
            # Some backends (e.g., SDPA) can skip returning attentions even when requested.
            vision_rollout_map = vector_to_patch_grid(image_hidden.norm(dim=-1), patch_grid)

        if getattr(text_out, "attentions", None):
            text_rollout = attention_rollout(text_out.attentions)[0]
            text_last_mean = last_layer_mean_attention(text_out.attentions)[0]
        else:
            text_len = int(text_out.last_hidden_state.shape[1])
            text_rollout = torch.eye(text_len, device=text_out.last_hidden_state.device)
            text_last_mean = text_rollout

        text_hidden = text_out.last_hidden_state[0]

        text_proj = self.model.text_projection(text_hidden)
        image_proj = self.model.visual_projection(image_hidden)

        align = cosine_similarity_matrix(text_proj, image_proj)
        token_scores = token_contribution_scores(align)

        token_global = (l2_normalize(text_proj, dim=-1) @ l2_normalize(outputs.image_embeds[0], dim=-1))

        grad_map = None
        if compute_gradients:
            image_full_hidden = vision_out.last_hidden_state
            image_full_hidden.retain_grad()
            objective = outputs.logits_per_image[0, 0]
            objective.backward()
            grads = image_full_hidden.grad[0, 1 : 1 + num_patches].norm(dim=-1)
            grad_map = vector_to_patch_grid(grads, patch_grid)

        attention_maps: dict[str, np.ndarray] = {
            "vision_rollout": vision_rollout_map,
            "text_rollout": to_numpy(text_rollout),
            "text_last_mean": to_numpy(text_last_mean),
        }
        if grad_map is not None:
            attention_maps["vision_grad"] = grad_map

        return AnalysisResult(
            model_family=self.family,
            model_name=self.model_name,
            prompt=prompt,
            tokens=batch.tokens,
            image_size=(image.width, image.height),
            patch_grid=patch_grid,
            global_score=float(outputs.logits_per_image[0, 0].item()),
            token_scores=token_scores,
            alignment_matrix=align,
            attention_maps=attention_maps,
            metadata={
                "token_to_global_similarity": to_numpy(token_global),
                "num_patches": num_patches,
            },
        )

    def score(self, image: Image.Image, prompt: str) -> float:
        self.ensure_loaded()
        assert self.model is not None

        batch = self.prepare(image, prompt)
        with torch.no_grad():
            outputs = self.model(**batch.model_inputs, return_dict=True)
        return float(outputs.logits_per_image[0, 0].item())
