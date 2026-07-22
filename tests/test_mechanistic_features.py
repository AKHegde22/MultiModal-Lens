from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

torch = pytest.importorskip("torch")

from multimodallens.adapters.base import ModelAdapter  # noqa: E402
from multimodallens.analysis.activation_patching import run_cross_modal_activation_patch  # noqa: E402
from multimodallens.analysis.circuits import discover_grounding_heads  # noqa: E402
from multimodallens.analysis.hooking import capture_layer_activations, list_hookable_layers  # noqa: E402
from multimodallens.analysis.logit_lens import run_multimodal_logit_lens  # noqa: E402
from multimodallens.types import AdapterBatch, AnalysisResult  # noqa: E402


class _ToyTokenizer:
    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        return [f"tok_{i}" for i in token_ids]


class _ToyLayer(torch.nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.proj(x))


class _ToyVLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 64, d_model: int = 16, num_layers: int = 4) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        self.image_proj = torch.nn.Linear(3, d_model)
        self.layers = torch.nn.ModuleList([_ToyLayer(d_model) for _ in range(num_layers)])
        self.lm_head = torch.nn.Linear(d_model, vocab_size)

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        output_attentions: bool = True,
        output_hidden_states: bool = True,
        return_dict: bool = True,
    ) -> SimpleNamespace:
        del output_attentions, output_hidden_states, return_dict

        text = self.embed(input_ids)

        mean = pixel_values.mean(dim=(-1, -2)).mean(dim=1)
        std = pixel_values.std(dim=(-1, -2)).mean(dim=1)
        max_v = pixel_values.amax(dim=(-1, -2)).mean(dim=1)
        image_feat = torch.stack([mean, std, max_v], dim=-1)
        image_token = self.image_proj(image_feat).unsqueeze(1)

        hidden = torch.cat([image_token, text], dim=1)
        hidden_states: list[torch.Tensor] = [hidden]
        attentions: list[torch.Tensor] = []

        for layer in self.layers:
            hidden = layer(hidden)
            hidden_states.append(hidden)

            scores = hidden @ hidden.transpose(-1, -2)
            attn = torch.softmax(scores, dim=-1)
            heads = attn.unsqueeze(1).repeat(1, 2, 1, 1)
            attentions.append(heads)

        logits = self.lm_head(hidden)
        return SimpleNamespace(
            logits=logits,
            hidden_states=tuple(hidden_states),
            attentions=tuple(attentions),
            image_hidden_states=image_token,
        )


class _ToyAdapter(ModelAdapter):
    family = "llava"

    def load(self) -> None:
        torch.manual_seed(7)
        self.model = _ToyVLM()
        self.tokenizer = _ToyTokenizer()
        self.processor = object()
        self.model.to(self.device)
        self.model.eval()

    def prepare(self, image: Image.Image, prompt: str) -> AdapterBatch:
        self.ensure_loaded()

        words = [w for w in prompt.strip().split() if w]
        if not words:
            words = ["empty"]

        token_ids = [(sum(ord(c) for c in w) % 50) + 1 for w in words]
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
            return self.model(**model_inputs, **self._forward_kwargs())
        with torch.no_grad():
            return self.model(**model_inputs, **self._forward_kwargs())

    def _segment_hidden_states(
        self,
        outputs: SimpleNamespace,
        batch: AdapterBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, list[str], list[int]]:
        hidden = outputs.hidden_states[-1][0]
        text_hidden = hidden[1:]
        image_hidden = hidden[:1]
        return text_hidden, image_hidden, batch.tokens, [0]

    def _score_outputs(self, outputs: SimpleNamespace, model_inputs: dict[str, torch.Tensor]) -> float:
        logits = outputs.logits
        input_ids = model_inputs["input_ids"]
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
        del compute_gradients
        batch = self.prepare(image, prompt)
        outputs = self._forward(batch.model_inputs, requires_grad=False)

        text_hidden, image_hidden, text_tokens, image_idx = self._segment_hidden_states(outputs, batch)
        align = (text_hidden @ image_hidden.transpose(-1, -2)).detach().cpu().numpy()
        token_scores = align.max(axis=1)

        rollout = outputs.attentions[-1][0].mean(dim=0)
        vision_score = float(rollout[-1, image_idx[0]].item())

        return AnalysisResult(
            model_family=self.family,
            model_name=self.model_name,
            prompt=prompt,
            tokens=text_tokens,
            image_size=(image.width, image.height),
            patch_grid=(1, 1),
            global_score=self._score_outputs(outputs, batch.model_inputs),
            token_scores=token_scores,
            alignment_matrix=align,
            attention_maps={"vision_rollout": np.array([[vision_score]], dtype=np.float32)},
        )

    def score(self, image: Image.Image, prompt: str) -> float:
        batch = self.prepare(image, prompt)
        outputs = self._forward(batch.model_inputs, requires_grad=False)
        return self._score_outputs(outputs, batch.model_inputs)


class _ToyClipAdapter(_ToyAdapter):
    family = "clip"


def _make_image(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = (rng.uniform(0, 255, size=(24, 24, 3))).astype(np.uint8)
    return Image.fromarray(arr)


def test_hook_layer_discovery_and_capture():
    adapter = _ToyAdapter(model_name="toy", device="cpu", dtype="float32")
    image = _make_image(0)

    layers = list_hookable_layers(adapter)
    assert len(layers) >= 4
    assert any("layers.0" in l for l in layers)

    run = capture_layer_activations(adapter=adapter, image=image, prompt="hello world")
    assert len(run.layers) >= 4
    assert run.layers[0].values.ndim == 3



def test_cross_modal_activation_patch_changes_score():
    adapter = _ToyAdapter(model_name="toy", device="cpu", dtype="float32")
    src = _make_image(1)
    tgt = _make_image(2)

    result = run_cross_modal_activation_patch(
        adapter=adapter,
        source_image=src,
        target_image=tgt,
        prompt="what is here",
        layer_index=1,
        visual_only=True,
    )

    assert result.patched_fraction > 0.0
    assert result.patched_score != result.baseline_score


def test_multimodal_logit_lens_returns_steps():
    adapter = _ToyAdapter(model_name="toy", device="cpu", dtype="float32")
    image = _make_image(3)

    out = run_multimodal_logit_lens(
        adapter=adapter,
        image=image,
        prompt="describe object",
        positions=[-1],
        top_k=3,
    )

    assert out.steps
    assert len(out.steps[0].top_tokens) == 3
    assert len(out.steps[0].top_probabilities) == 3


def test_grounding_head_discovery_returns_ranked_heads():
    adapter = _ToyAdapter(model_name="toy", device="cpu", dtype="float32")
    image = _make_image(4)

    out = discover_grounding_heads(
        adapter=adapter,
        image=image,
        prompt="count the objects",
        mask_fraction=1.0,
        top_k=5,
    )

    assert out.heads
    assert out.heads[0].grounding_score >= 0.0


def test_grounding_head_discovery_supports_clip_family():
    adapter = _ToyClipAdapter(model_name="toy", device="cpu", dtype="float32")
    image = _make_image(5)

    out = discover_grounding_heads(
        adapter=adapter,
        image=image,
        prompt="locate objects",
        mask_fraction=0.5,
        top_k=5,
    )

    assert out.heads


def test_logit_lens_falls_back_to_input_embeddings():
    adapter = _ToyClipAdapter(model_name="toy", device="cpu", dtype="float32")
    adapter.ensure_loaded()
    assert adapter.model is not None

    # Force fallback path by disabling output embeddings and exposing input embeddings.
    adapter.model.get_output_embeddings = lambda: None  # type: ignore[method-assign]
    adapter.model.get_input_embeddings = lambda: adapter.model.embed  # type: ignore[method-assign]

    out = run_multimodal_logit_lens(
        adapter=adapter,
        image=_make_image(6),
        prompt="describe object",
        positions=[-1],
        top_k=3,
    )

    assert out.steps


def test_logit_lens_handles_clip_not_implemented_get_input_embeddings():
    adapter = _ToyClipAdapter(model_name="toy", device="cpu", dtype="float32")
    adapter.ensure_loaded()
    assert adapter.model is not None

    adapter.model.get_output_embeddings = lambda: None  # type: ignore[method-assign]

    def _raise_not_impl():
        raise NotImplementedError("get_input_embeddings is not implemented")

    adapter.model.get_input_embeddings = _raise_not_impl  # type: ignore[method-assign]
    adapter.model.text_model = SimpleNamespace(  # type: ignore[attr-defined]
        embeddings=SimpleNamespace(token_embedding=adapter.model.embed)
    )

    out = run_multimodal_logit_lens(
        adapter=adapter,
        image=_make_image(7),
        prompt="describe object",
        positions=[-1],
        top_k=3,
    )

    assert out.steps


def test_logit_lens_projects_clip_hidden_states_for_embedding_similarity():
    adapter = _ToyClipAdapter(model_name="toy", device="cpu", dtype="float32")
    adapter.ensure_loaded()
    assert adapter.model is not None

    proj_dim = 8
    adapter.model.get_output_embeddings = lambda: None  # type: ignore[method-assign]

    def _raise_not_impl():
        raise NotImplementedError("get_input_embeddings is not implemented")

    adapter.model.get_input_embeddings = _raise_not_impl  # type: ignore[method-assign]
    adapter.model.text_model = SimpleNamespace(  # type: ignore[attr-defined]
        embeddings=SimpleNamespace(token_embedding=torch.nn.Embedding(64, proj_dim))
    )
    adapter.model.visual_projection = torch.nn.Linear(16, proj_dim, bias=False)  # type: ignore[attr-defined]

    out = run_multimodal_logit_lens(
        adapter=adapter,
        image=_make_image(8),
        prompt="describe object",
        positions=[-1],
        top_k=3,
    )

    assert out.steps