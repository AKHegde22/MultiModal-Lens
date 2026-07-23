import pytest
import torch
from types import SimpleNamespace
from PIL import Image
import numpy as np
from typing import Any

from multimodallens.adapters.base import ModelAdapter
from multimodallens.types import AdapterBatch
from multimodallens.analysis.dla import run_multimodal_dla, DLAResult

class _DeterministicSelfAttn(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.v_proj(x)
        return self.o_proj(v)

class _DeterministicLayer(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.self_attn = _DeterministicSelfAttn(hidden_size)
        self.mlp = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(x)
        x = x + self.mlp(x)
        return x

class _DeterministicVLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 16, hidden_size: int = 16, num_heads: int = 2, head_dim: int = 8):
        super().__init__()
        self.config = SimpleNamespace(
            num_attention_heads=num_heads,
            hidden_size=hidden_size,
            head_dim=head_dim,
            vocab_size=vocab_size
        )
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.image_proj = torch.nn.Linear(3, hidden_size, bias=False)
        self.layers = torch.nn.ModuleList([
            _DeterministicLayer(hidden_size) for _ in range(2)
        ])
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        **kwargs: Any
    ) -> SimpleNamespace:
        text = self.embed(input_ids)
        
        mean = pixel_values.mean(dim=(-1, -2))
        image_token = self.image_proj(mean).unsqueeze(1)
        
        hidden = torch.cat([image_token, text], dim=1)
        
        for layer in self.layers:
            hidden = layer(hidden)
            
        logits = self.lm_head(hidden)
        return SimpleNamespace(logits=logits)

class _ToyTokenizer:
    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        return [f"tok_{i}" for i in token_ids]
    
    def convert_tokens_to_ids(self, token: str) -> int:
        return int(token.split("_")[-1])

class _DeterministicAdapter(ModelAdapter):
    family = "llava"

    def load(self) -> None:
        torch.manual_seed(42)
        self.model = _DeterministicVLM()
        self.tokenizer = _ToyTokenizer()
        self.processor = object()
        self.model.to(self.device)
        self.model.eval()

    def prepare(self, image: Image.Image, prompt: str) -> AdapterBatch:
        self.ensure_loaded()

        words = [w for w in prompt.strip().split() if w]
        if not words:
            words = ["empty"]

        token_ids = [(sum(ord(c) for c in w) % 15) + 1 for w in words]
        tokens = [f"tok_{i}" for i in token_ids]

        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        pixel_values = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        input_ids = torch.tensor([token_ids], dtype=torch.long)

        return AdapterBatch(
            model_inputs=self._move_inputs(
                {
                    "input_ids": input_ids,
                    "pixel_values": pixel_values,
                }
            ),
            tokens=tokens,
            token_ids=token_ids,
        )

    def _forward(self, model_inputs: dict[str, torch.Tensor], requires_grad: bool) -> SimpleNamespace:
        assert self.model is not None
        if requires_grad:
            self.model.zero_grad(set_to_none=True)
            return self.model(**model_inputs)
        with torch.no_grad():
            return self.model(**model_inputs)
            
    def analyze(self, image: Image.Image, prompt: str, compute_gradients: bool = False) -> Any:
        return None

    def score(self, image: Image.Image, prompt: str) -> float:
        return 0.0

@pytest.fixture
def adapter():
    return _DeterministicAdapter(model_name="toy", device="cpu", dtype="float32")

@pytest.fixture
def image():
    return Image.new('RGB', (32, 32), 'red')

def test_dla_decomposition_is_exhaustive(adapter, image):
    res = run_multimodal_dla(adapter, image, "hello", target_token="tok_1")
    assert abs(res.residual_error) < 1e-4

def test_dla_returns_per_head_contributions(adapter, image):
    res = run_multimodal_dla(adapter, image, "hello", target_token="tok_1")
    assert len(res.head_contributions) == 4  # 2 layers * 2 heads
    layers_heads = {(hc.layer, hc.head) for hc in res.head_contributions}
    assert layers_heads == {(0, 0), (0, 1), (1, 0), (1, 1)}
    for hc in res.head_contributions:
        assert hc.layer_name.startswith(f"layers.{hc.layer}.self_attn.o_proj")

def test_dla_returns_mlp_contributions(adapter, image):
    res = run_multimodal_dla(adapter, image, "hello", target_token="tok_1")
    assert len(res.mlp_contributions) == 2
    layers = {mc.layer for mc in res.mlp_contributions}
    assert layers == {0, 1}
    for mc in res.mlp_contributions:
        assert mc.layer_name.startswith(f"layers.{mc.layer}.mlp")

def test_dla_head_count_matches_model(adapter, image):
    res = run_multimodal_dla(adapter, image, "hello", target_token="tok_1")
    assert len(res.head_contributions) == 4

def test_dla_result_backward_compatible(adapter, image):
    res = run_multimodal_dla(adapter, image, "hello", target_token="tok_1")
    contribs = res.contributions
    assert len(contribs) == len(res.head_contributions) + len(res.mlp_contributions) + 1
    # embedding should be present
    assert any(c.layer_name == "embedding" for c in contribs)
