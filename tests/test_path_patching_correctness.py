import pytest
import torch
from types import SimpleNamespace
from PIL import Image
import numpy as np
from typing import Any

from multimodallens.adapters.base import ModelAdapter
from multimodallens.types import AdapterBatch
from multimodallens.analysis.path_patching import run_causal_path_patching, EdgeEffect, PathPatchingResult

class _DeterministicSelfAttn(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (x.shape[-1] ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        return self.o_proj(out)

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

        # Use deterministic image based on prompt length
        arr = np.ones((32, 32, 3), dtype=np.float32) * len(prompt)
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
def clean_image():
    return Image.new('RGB', (32, 32), 'red')

@pytest.fixture
def corrupt_image():
    return Image.new('RGB', (32, 32), 'blue')

def test_path_patching_produces_edge_effects(adapter, clean_image, corrupt_image):
    res = run_causal_path_patching(
        adapter=adapter,
        clean_image=clean_image,
        clean_prompt="clean prompt",
        corrupted_image=corrupt_image,
        corrupted_prompt="corrupted prompt",
        sender_layers=["layers.0"],
        receiver_layers=["layers.1"],
    )
    assert isinstance(res, PathPatchingResult)
    assert len(res.effects) > 0
    for effect in res.effects:
        assert isinstance(effect, EdgeEffect)
        assert effect.sender_layer == "layers.0"
        assert effect.receiver_layer == "layers.1"
        assert hasattr(effect, "receiver_channel")
        assert effect.receiver_channel in ["q", "k", "v"]
        assert hasattr(effect, "causal_effect")

def test_path_patching_channels_are_independent(adapter, clean_image, corrupt_image):
    # Patch q
    res_q = run_causal_path_patching(
        adapter=adapter,
        clean_image=clean_image,
        clean_prompt="clean prompt",
        corrupted_image=corrupt_image,
        corrupted_prompt="corrupted prompt",
        sender_layers=["layers.0"],
        receiver_layers=["layers.1"],
        receiver_channels=["q"]
    )
    # Patch k
    res_k = run_causal_path_patching(
        adapter=adapter,
        clean_image=clean_image,
        clean_prompt="clean prompt",
        corrupted_image=corrupt_image,
        corrupted_prompt="corrupted prompt",
        sender_layers=["layers.0"],
        receiver_layers=["layers.1"],
        receiver_channels=["k"]
    )
    
    assert len(res_q.effects) == 1
    assert len(res_k.effects) == 1
    effect_q = res_q.effects[0]
    effect_k = res_k.effects[0]
    
    # Check that they patched different channels and produced different results
    assert effect_q.receiver_channel == "q"
    assert effect_k.receiver_channel == "k"
    # Given the deterministic random init of linear layers in _DeterministicVLM, they should be different
    assert effect_q.causal_effect != effect_k.causal_effect

def test_path_patching_uses_forward_input_patcher(adapter, clean_image, corrupt_image):
    # If it patches the input of the receiver, different receivers should yield different scores
    res_q1 = run_causal_path_patching(
        adapter=adapter,
        clean_image=clean_image,
        clean_prompt="clean prompt",
        corrupted_image=corrupt_image,
        corrupted_prompt="corrupted prompt",
        sender_layers=["layers.0"],
        receiver_layers=["layers.1"],
        receiver_channels=["q"]
    )
    
    # Now wait, we only have 2 layers (0 and 1)
    # Let's patch receiver layers.1.mlp instead of layers.1.self_attn
    res_mlp = run_causal_path_patching(
        adapter=adapter,
        clean_image=clean_image,
        clean_prompt="clean prompt",
        corrupted_image=corrupt_image,
        corrupted_prompt="corrupted prompt",
        sender_layers=["layers.0"],
        receiver_layers=["layers.1.mlp"],
        receiver_channels=["residual"]
    )
    
    eff1 = res_q1.effects[0]
    eff2 = res_mlp.effects[0]
    
    assert eff1.receiver_layer == "layers.1"
    assert eff2.receiver_layer == "layers.1.mlp"
    assert eff1.causal_effect != eff2.causal_effect

def test_path_patching_backward_compatible_result(adapter, clean_image, corrupt_image):
    res = run_causal_path_patching(
        adapter=adapter,
        clean_image=clean_image,
        clean_prompt="clean prompt",
        corrupted_image=corrupt_image,
        corrupted_prompt="corrupted prompt",
        sender_layers=["layers.0"],
        receiver_layers=["layers.1"],
        receiver_channels=["q"]
    )
    
    d = res.to_dict()
    assert "effects" in d
    assert len(d["effects"]) > 0
    eff = d["effects"][0]
    assert "receiver_channel" in eff
    assert "causal_effect" in eff
    assert "sender_layer" in eff
