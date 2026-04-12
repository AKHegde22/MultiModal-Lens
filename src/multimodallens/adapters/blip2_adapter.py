"""Adapter implementation for BLIP-2 / Q-Former based models."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration

from multimodallens.adapters.base import ModelAdapter
from multimodallens.analysis.alignment import cosine_similarity_matrix, token_contribution_scores
from multimodallens.analysis.attention import attention_rollout, vector_to_patch_grid
from multimodallens.types import AnalysisResult, AdapterBatch
from multimodallens.utils.image_ops import infer_patch_layout
from multimodallens.utils.tensor_ops import to_numpy


class BLIP2Adapter(ModelAdapter):
    """Adapter for BLIP-2 style models exposing vision + Q-Former internals."""

    family = "blip2"

    def load(self) -> None:
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        self.tokenizer = getattr(self.processor, "tokenizer", None)

        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype if self.device.type != "cpu" else torch.float32,
            trust_remote_code=self.trust_remote_code,
            low_cpu_mem_usage=self.low_cpu_mem_usage,
        )
        self.model.to(self.device)  # type: ignore[arg-type]
        self.model.eval()

    def prepare(self, image: Image.Image, prompt: str) -> AdapterBatch:
        self.ensure_loaded()
        assert self.processor is not None

        packed = self.processor(images=image, text=prompt, return_tensors="pt")
        token_ids = packed.get("input_ids", torch.empty(0, dtype=torch.long)).tolist()
        token_ids = token_ids[0] if token_ids else []

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

    def _extract_text_hidden(self, outputs: Any, seq_len_hint: int) -> torch.Tensor:
        if hasattr(outputs, "language_model_outputs") and outputs.language_model_outputs is not None:
            lm = outputs.language_model_outputs
            hidden_states = getattr(lm, "hidden_states", None)
            if hidden_states:
                hidden = hidden_states[-1][0]
                if seq_len_hint > 0 and hidden.shape[0] >= seq_len_hint:
                    return hidden[-seq_len_hint:]
                return hidden

        if hasattr(outputs, "qformer_outputs") and outputs.qformer_outputs is not None:
            qf = outputs.qformer_outputs
            hidden_states = getattr(qf, "hidden_states", None)
            if hidden_states:
                return hidden_states[-1][0]

        if hasattr(outputs, "hidden_states") and outputs.hidden_states:
            return outputs.hidden_states[-1][0]

        raise RuntimeError("Unable to extract text-side hidden states from BLIP-2 outputs.")

    def _extract_vision_outputs(self, outputs: Any) -> Any:
        vis = getattr(outputs, "vision_outputs", None)
        if vis is None:
            vis = getattr(outputs, "vision_model_output", None)
        if vis is None:
            raise RuntimeError("BLIP-2 outputs do not expose vision outputs.")
        return vis

    def _sequence_score(self, outputs: Any, model_inputs: dict[str, Any]) -> float:
        logits = getattr(outputs, "logits", None)
        input_ids = model_inputs.get("input_ids")

        if logits is None:
            raise RuntimeError("BLIP-2 outputs missing logits; cannot compute perturbation score.")

        if input_ids is None or logits.shape[1] < 2:
            return float(logits[0, -1].max().item())

        labels = input_ids[:, 1:]
        pred = logits[:, : labels.shape[1], :]
        log_probs = pred.log_softmax(dim=-1)
        token_logp = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        return float(token_logp.mean().item())

    def analyze(
        self,
        image: Image.Image,
        prompt: str,
        compute_gradients: bool = False,
    ) -> AnalysisResult:
        self.ensure_loaded()

        batch = self.prepare(image, prompt)
        outputs = self._forward(batch.model_inputs, requires_grad=compute_gradients)

        vision = self._extract_vision_outputs(outputs)
        vision_hidden_full = vision.last_hidden_state
        if vision_hidden_full.shape[1] > 1:
            image_hidden = vision_hidden_full[0, 1:]
            image_start_idx = 1
        else:
            image_hidden = vision_hidden_full[0]
            image_start_idx = 0

        num_patches = int(image_hidden.shape[0])
        layout = infer_patch_layout(num_patches)
        patch_grid = (layout.rows, layout.cols)

        if getattr(vision, "attentions", None):
            rollout = attention_rollout(vision.attentions)[0]
            cls_to_patches = rollout[0, image_start_idx : image_start_idx + num_patches]
            vision_rollout_map = vector_to_patch_grid(cls_to_patches, patch_grid)
        else:
            fallback = image_hidden.norm(dim=-1)
            vision_rollout_map = vector_to_patch_grid(fallback, patch_grid)

        text_hidden = self._extract_text_hidden(outputs, seq_len_hint=len(batch.tokens))
        text_len = text_hidden.shape[0]

        if not batch.tokens:
            tokens = [f"tok_{i}" for i in range(text_len)]
        elif len(batch.tokens) > text_len:
            tokens = batch.tokens[-text_len:]
        elif len(batch.tokens) < text_len:
            tokens = [f"h_{i}" for i in range(text_len - len(batch.tokens))] + batch.tokens
        else:
            tokens = batch.tokens

        d = min(text_hidden.shape[-1], image_hidden.shape[-1])
        align = cosine_similarity_matrix(text_hidden[:, :d], image_hidden[:, :d])
        token_scores = token_contribution_scores(align)

        grad_map = None
        if compute_gradients:
            vision_hidden_full.retain_grad()
            objective = getattr(outputs, "logits")[0, -1].max()
            objective.backward()
            grads = vision_hidden_full.grad[0, image_start_idx : image_start_idx + num_patches].norm(dim=-1)
            grad_map = vector_to_patch_grid(grads, patch_grid)

        attention_maps: dict[str, np.ndarray] = {
            "vision_rollout": vision_rollout_map,
        }
        qformer = getattr(outputs, "qformer_outputs", None)
        if qformer is not None and getattr(qformer, "cross_attentions", None):
            cross = qformer.cross_attentions[-1][0].mean(dim=0)
            attention_maps["qformer_cross_last"] = to_numpy(cross)
        if grad_map is not None:
            attention_maps["vision_grad"] = grad_map

        return AnalysisResult(
            model_family=self.family,
            model_name=self.model_name,
            prompt=prompt,
            tokens=tokens,
            image_size=(image.width, image.height),
            patch_grid=patch_grid,
            global_score=self._sequence_score(outputs, batch.model_inputs),
            token_scores=token_scores,
            alignment_matrix=align,
            attention_maps=attention_maps,
            metadata={
                "num_patches": num_patches,
                "text_hidden_len": int(text_len),
            },
        )

    def score(self, image: Image.Image, prompt: str) -> float:
        self.ensure_loaded()
        batch = self.prepare(image, prompt)
        outputs = self._forward(batch.model_inputs, requires_grad=False)
        return self._sequence_score(outputs, batch.model_inputs)
