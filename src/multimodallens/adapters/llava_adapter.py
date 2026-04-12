"""Adapter implementation for LLaVA-style interleaved decoder models."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
import transformers
from transformers import AutoProcessor

from multimodallens.adapters.base import ModelAdapter
from multimodallens.analysis.alignment import cosine_similarity_matrix, token_contribution_scores
from multimodallens.analysis.attention import attention_rollout, vector_to_patch_grid
from multimodallens.types import AnalysisResult, AdapterBatch
from multimodallens.utils.image_ops import infer_patch_layout
from multimodallens.utils.tensor_ops import to_numpy


class LlavaAdapter(ModelAdapter):
    """Adapter for LLaVA/Qwen-VL-like decoder-only VLMs."""

    family = "llava"

    def load(self) -> None:
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        self.tokenizer = getattr(self.processor, "tokenizer", None)

        model_cls: Any = getattr(transformers, "LlavaForConditionalGeneration", None)
        if model_cls is None:
            model_cls = getattr(transformers, "AutoModelForImageTextToText", None)
        if model_cls is None:
            model_cls = getattr(transformers, "AutoModelForVision2Seq", None)
        if model_cls is None:
            raise RuntimeError(
                "No compatible vision-language auto model class found in transformers. "
                "Expected one of: LlavaForConditionalGeneration, AutoModelForImageTextToText, "
                "AutoModelForVision2Seq."
            )

        self.model = model_cls.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype if self.device.type != "cpu" else torch.float32,
            trust_remote_code=self.trust_remote_code,
            low_cpu_mem_usage=self.low_cpu_mem_usage,
            attn_implementation="eager",
        )
        self.model.to(self.device)  # type: ignore[arg-type]
        self.model.eval()

    def _format_prompt(self, prompt: str) -> str:
        assert self.processor is not None
        if hasattr(self.processor, "apply_chat_template"):
            conv = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            try:
                return self.processor.apply_chat_template(
                    conv,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except TypeError:
                return self.processor.apply_chat_template(conv, add_generation_prompt=True)
        return f"<image>\n{prompt}"

    def prepare(self, image: Image.Image, prompt: str) -> AdapterBatch:
        self.ensure_loaded()
        assert self.processor is not None

        formatted = self._format_prompt(prompt)
        packed = self.processor(text=formatted, images=image, return_tensors="pt")

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

    def _image_token_index(self) -> int | None:
        assert self.model is not None
        idx = getattr(self.model.config, "image_token_index", None)
        if idx is not None:
            return int(idx)
        if self.tokenizer is not None:
            maybe = self.tokenizer.convert_tokens_to_ids("<image>")
            if maybe is not None and maybe >= 0:
                return int(maybe)
        return None

    def _extract_image_hidden(self, outputs: Any) -> torch.Tensor | None:
        image_hidden = getattr(outputs, "image_hidden_states", None)
        if image_hidden is None:
            return None

        if isinstance(image_hidden, (tuple, list)):
            image_hidden = image_hidden[-1]

        if image_hidden.ndim == 4:
            # [B, num_images, num_patches, D]
            return image_hidden[0].reshape(-1, image_hidden.shape[-1])
        if image_hidden.ndim == 3:
            # [B, num_patches, D]
            return image_hidden[0]
        return None

    def _segment_hidden_states(
        self,
        outputs: Any,
        batch: AdapterBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, list[str], list[int]]:
        """Segment final hidden states into text/image regions.

        Returns:
            text_hidden, image_hidden, text_tokens, image_seq_indices
        """
        hidden = outputs.hidden_states[-1][0]
        seq_len = hidden.shape[0]
        input_len = len(batch.token_ids)
        image_token_idx = self._image_token_index()

        image_positions: list[int] = []
        if image_token_idx is not None and batch.token_ids:
            image_positions = [i for i, t in enumerate(batch.token_ids) if t == image_token_idx]

        image_hidden = self._extract_image_hidden(outputs)

        if image_hidden is not None:
            # Derive text hidden from token positions excluding explicit image placeholders.
            if batch.tokens:
                text_positions = [i for i in range(min(input_len, seq_len)) if i not in image_positions]
                if text_positions:
                    text_hidden = hidden[text_positions]
                    text_tokens = [batch.tokens[i] for i in text_positions]
                else:
                    text_hidden = hidden
                    text_tokens = [f"tok_{i}" for i in range(seq_len)]
            else:
                text_hidden = hidden
                text_tokens = [f"tok_{i}" for i in range(seq_len)]

            image_seq_indices: list[int] = []
            if image_positions and seq_len > input_len:
                expanded = seq_len - input_len + len(image_positions)
                start = image_positions[0]
                end = min(start + expanded, seq_len)
                image_seq_indices = list(range(start, end))
            elif image_positions:
                image_seq_indices = image_positions

            return text_hidden, image_hidden, text_tokens, image_seq_indices

        # Fallback when model output does not expose image_hidden_states.
        if image_positions and seq_len > input_len:
            expanded = seq_len - input_len + len(image_positions)
            start = image_positions[0]
            end = min(start + expanded, seq_len)
            image_seq_indices = list(range(start, end))
            image_hidden = hidden[start:end]
            text_hidden = torch.cat([hidden[:start], hidden[end:]], dim=0)
            text_tokens = [tok for i, tok in enumerate(batch.tokens) if i not in image_positions]
            return text_hidden, image_hidden, text_tokens, image_seq_indices

        if image_positions:
            image_hidden = hidden[image_positions]
            keep = [i for i in range(seq_len) if i not in image_positions]
            text_hidden = hidden[keep]
            text_tokens = [batch.tokens[i] for i in range(len(batch.tokens)) if i not in image_positions]
            return text_hidden, image_hidden, text_tokens, image_positions

        # Final fallback: infer a visual prefix by model vision config patch count.
        guessed = self._guess_patch_count()
        guessed = min(max(1, guessed), max(1, seq_len // 2))
        image_hidden = hidden[:guessed]
        text_hidden = hidden[guessed:]

        if batch.tokens and len(batch.tokens) >= text_hidden.shape[0]:
            text_tokens = batch.tokens[-text_hidden.shape[0] :]
        else:
            text_tokens = [f"tok_{i}" for i in range(text_hidden.shape[0])]

        return text_hidden, image_hidden, text_tokens, list(range(guessed))

    def _guess_patch_count(self) -> int:
        assert self.model is not None
        vis_cfg = getattr(self.model.config, "vision_config", None)
        if vis_cfg is None:
            return 256
        image_size = int(getattr(vis_cfg, "image_size", 336))
        patch = int(getattr(vis_cfg, "patch_size", 14))
        return max(1, (image_size // patch) ** 2)

    def _sequence_score(self, outputs: Any, model_inputs: dict[str, Any]) -> float:
        logits = getattr(outputs, "logits", None)
        input_ids = model_inputs.get("input_ids")

        if logits is None:
            raise RuntimeError("LLaVA outputs missing logits; cannot compute perturbation score.")

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

        text_hidden, image_hidden, text_tokens, image_seq_indices = self._segment_hidden_states(outputs, batch)

        if text_hidden.numel() == 0 or image_hidden.numel() == 0:
            raise RuntimeError("Could not extract non-empty text/image hidden states for LLaVA analysis.")

        num_patches = int(image_hidden.shape[0])
        layout = infer_patch_layout(num_patches)
        patch_grid = (layout.rows, layout.cols)

        d = min(text_hidden.shape[-1], image_hidden.shape[-1])
        align = cosine_similarity_matrix(text_hidden[:, :d], image_hidden[:, :d])
        token_scores = token_contribution_scores(align)

        attention_maps: dict[str, np.ndarray] = {}

        vision_rollout_map = None
        if getattr(outputs, "attentions", None):
            rollout = attention_rollout(outputs.attentions)[0]
            query_idx = rollout.shape[0] - 1

            if image_seq_indices:
                clipped = [i for i in image_seq_indices if i < rollout.shape[-1]]
                if clipped:
                    patch_attn = rollout[query_idx, clipped]
                    if patch_attn.shape[0] == num_patches:
                        vision_rollout_map = vector_to_patch_grid(patch_attn, patch_grid)
            attention_maps["decoder_rollout"] = to_numpy(rollout)

        if vision_rollout_map is None:
            vision_rollout_map = vector_to_patch_grid(image_hidden.norm(dim=-1), patch_grid)

        attention_maps["vision_rollout"] = vision_rollout_map

        grad_map = None
        if compute_gradients:
            hidden_last = outputs.hidden_states[-1]
            hidden_last.retain_grad()
            objective = outputs.logits[0, -1].max()
            objective.backward()

            grads = hidden_last.grad[0]
            if image_seq_indices:
                clipped = [i for i in image_seq_indices if i < grads.shape[0]]
                if clipped:
                    grad_vec = grads[clipped].norm(dim=-1)
                else:
                    grad_vec = image_hidden.norm(dim=-1)
            else:
                grad_vec = image_hidden.norm(dim=-1)

            if grad_vec.shape[0] != num_patches:
                grad_vec = image_hidden.norm(dim=-1)
            grad_map = vector_to_patch_grid(grad_vec, patch_grid)
            attention_maps["vision_grad"] = grad_map

        if len(text_tokens) > text_hidden.shape[0]:
            text_tokens = text_tokens[-text_hidden.shape[0] :]
        elif len(text_tokens) < text_hidden.shape[0]:
            prefix = [f"h_{i}" for i in range(text_hidden.shape[0] - len(text_tokens))]
            text_tokens = prefix + text_tokens

        return AnalysisResult(
            model_family=self.family,
            model_name=self.model_name,
            prompt=prompt,
            tokens=text_tokens,
            image_size=(image.width, image.height),
            patch_grid=patch_grid,
            global_score=self._sequence_score(outputs, batch.model_inputs),
            token_scores=token_scores,
            alignment_matrix=align,
            attention_maps=attention_maps,
            metadata={
                "num_patches": num_patches,
                "image_seq_indices": image_seq_indices,
            },
        )

    def score(self, image: Image.Image, prompt: str) -> float:
        self.ensure_loaded()
        batch = self.prepare(image, prompt)
        outputs = self._forward(batch.model_inputs, requires_grad=False)
        return self._sequence_score(outputs, batch.model_inputs)
