"""Base adapter interface for multimodal model families."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from PIL import Image

from multimodallens.types import AnalysisResult, AdapterBatch


DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_device(device: str) -> torch.device:
    """Resolve runtime device from requested string."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


class ModelAdapter(ABC):
    """Abstract interface for model-family-specific extraction."""

    family: str = "unknown"

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: str = "float16",
        trust_remote_code: bool = False,
        low_cpu_mem_usage: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.dtype_name = dtype
        self.torch_dtype = DTYPE_MAP.get(dtype, torch.float16)
        self.trust_remote_code = trust_remote_code
        self.low_cpu_mem_usage = low_cpu_mem_usage

        self.model: Any | None = None
        self.processor: Any | None = None
        self.tokenizer: Any | None = None

    @abstractmethod
    def load(self) -> None:
        """Load processor/tokenizer/model into memory."""

    @abstractmethod
    def prepare(self, image: Image.Image, prompt: str) -> AdapterBatch:
        """Prepare tensors and token metadata."""

    @abstractmethod
    def analyze(
        self,
        image: Image.Image,
        prompt: str,
        compute_gradients: bool = False,
    ) -> AnalysisResult:
        """Run full analysis and return normalized result."""

    @abstractmethod
    def score(self, image: Image.Image, prompt: str) -> float:
        """Scalar score for perturbation tests."""

    def ensure_loaded(self) -> None:
        """Lazy-load model if needed."""
        if self.model is None or self.processor is None:
            self.load()

    def _move_inputs(self, model_inputs: dict[str, Any]) -> dict[str, Any]:
        """Move tensor fields to runtime device."""
        moved: dict[str, Any] = {}
        for key, value in model_inputs.items():
            if torch.is_tensor(value):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def _forward_kwargs(self) -> dict[str, Any]:
        """Shared kwargs for model forward calls."""
        return {
            "output_attentions": True,
            "output_hidden_states": True,
            "return_dict": True,
        }
