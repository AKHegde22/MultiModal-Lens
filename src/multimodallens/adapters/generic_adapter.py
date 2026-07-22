"""Generic config-driven ModelAdapter for Vision-Language Models."""

from __future__ import annotations

from typing import Any, Sequence
import torch
from PIL import Image

from multimodallens.adapters.base import ModelAdapter
from multimodallens.core.config_schema import MultimodalConfig, get_multimodal_config
from multimodallens.types import AdapterBatch

ModelBatch = AdapterBatch



class GenericVLMAdapter(ModelAdapter):
    """Universal adapter leveraging MultimodalConfig for automated model routing."""

    def __init__(
        self,
        family: str,
        model_name: str,
        device: str = "auto",
        dtype: str = "float16",
        trust_remote_code: bool = False,
        low_cpu_mem_usage: bool = True,
        config: MultimodalConfig | None = None,
    ) -> None:
        self.config = config or get_multimodal_config(family)
        super().__init__(
            model_name=model_name,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
        self.family = family


    def load(self) -> None:
        model, processor = self.load_model_and_processor()
        self.model = model
        self.processor = processor
        if hasattr(processor, "tokenizer"):
            self.tokenizer = processor.tokenizer

    def load_model_and_processor(self) -> tuple[Any, Any]:
        from transformers import AutoProcessor, AutoModelForVision2Seq

        processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )

        model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            trust_remote_code=self.trust_remote_code,
            low_cpu_mem_usage=self.low_cpu_mem_usage,
        )

        model = model.to(self.device)
        return model, processor

    def analyze(
        self,
        image: Image.Image,
        prompt: str,
        compute_gradients: bool = False,
    ) -> AnalysisResult:
        import numpy as np
        from multimodallens.types import AnalysisResult

        batch = self.prepare(image, prompt)
        outputs = self._forward(batch.model_inputs, requires_grad=compute_gradients)
        logits = getattr(outputs, "logits", None)
        score_val = float(logits.max().item()) if torch.is_tensor(logits) else 0.0

        n_toks = len(batch.tokens) or 1
        return AnalysisResult(
            model_family=self.family,
            model_name=self.model_name,
            prompt=prompt,
            tokens=batch.tokens,
            image_size=image.size,
            patch_grid=(24, 24),
            global_score=score_val,
            token_scores=np.ones(n_toks, dtype=np.float32),
            alignment_matrix=np.ones((n_toks, 576), dtype=np.float32),
            attention_maps={"mean": np.ones((24, 24), dtype=np.float32)},
            metadata={"config": self.config.family},
        )

    def score(self, image: Image.Image, prompt: str) -> float:
        batch = self.prepare(image, prompt)
        outputs = self._forward(batch.model_inputs, requires_grad=False)
        logits = getattr(outputs, "logits", None)
        return float(logits.max().item()) if torch.is_tensor(logits) else 0.0


    def prepare(self, image: Image.Image, prompt: str) -> ModelBatch:
        self.ensure_loaded()
        assert self.processor is not None

        image_rgbs = [image.convert("RGB")]
        inputs = self.processor(
            text=prompt,
            images=image_rgbs,
            return_tensors="pt",
        )

        moved_inputs: dict[str, Any] = {}
        for k, v in inputs.items():
            if torch.is_tensor(v):
                if v.dtype in (torch.float32, torch.float16, torch.bfloat16):
                    moved_inputs[k] = v.to(device=self.device, dtype=self.torch_dtype)
                else:
                    moved_inputs[k] = v.to(device=self.device)
            else:
                moved_inputs[k] = v

        input_ids = moved_inputs.get(self.config.text_input_key)
        tokens: Sequence[str] = []
        ids: list[int] = []
        if input_ids is not None and torch.is_tensor(input_ids):
            ids = [int(i) for i in input_ids[0].detach().cpu().tolist()]
            if hasattr(self.processor, "tokenizer"):
                tokenizer = self.processor.tokenizer
                if hasattr(tokenizer, "convert_ids_to_tokens"):
                    tokens = [str(tok) for tok in tokenizer.convert_ids_to_tokens(ids)]
                else:
                    tokens = [str(i) for i in ids]

        return AdapterBatch(
            model_inputs=moved_inputs,
            tokens=list(tokens),
            token_ids=ids,
        )


    def _forward(self, model_inputs: dict[str, Any], requires_grad: bool = False) -> Any:
        self.ensure_loaded()
        assert self.model is not None

        if requires_grad:
            return self.model(**model_inputs)
        with torch.no_grad():
            return self.model(**model_inputs)

    def _segment_hidden_states(
        self,
        hidden_states: torch.Tensor,
        batch: ModelBatch,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        input_ids = batch.model_inputs.get(self.config.text_input_key)
        if input_ids is None or not torch.is_tensor(input_ids):
            return None, hidden_states

        token_seq = input_ids[0]
        tokenizer = getattr(self.processor, "tokenizer", None)
        image_token_id = self.config.image_token_id

        if image_token_id is None and tokenizer is not None:
            try:
                image_token_id = tokenizer.convert_tokens_to_ids(self.config.image_token_str)
            except Exception:
                image_token_id = None

        if image_token_id is not None:
            image_mask = token_seq == image_token_id
            text_mask = ~image_mask
            if hidden_states.ndim == 3:
                seq = hidden_states[0]
                vis = seq[image_mask] if image_mask.any() else None
                txt = seq[text_mask] if text_mask.any() else seq
                return vis, txt

        return None, hidden_states[0] if hidden_states.ndim == 3 else hidden_states
