from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from multimodallens.core.hook_point import HookPoint
from multimodallens.analysis.sae import SparseAutoencoder
from multimodallens.analysis.induction import detect_cross_modal_induction_heads
from multimodallens.analysis.neurons import analyze_neuron_activations, NeuronActivationSummary
from multimodallens.types import LayerActivationRun, LayerActivation


def test_hook_point_pass_through_and_intervention():
    hp = HookPoint(name="test_hook")
    assert hp.name == "test_hook"

    x = torch.tensor([1.0, 2.0, 3.0])
    out = hp(x)
    assert torch.equal(out, x)

    def add_one(t):
        return t + 1.0

    hp.add_hook(add_one)
    out_patched = hp(x)
    assert torch.equal(out_patched, torch.tensor([2.0, 3.0, 4.0]))

    hp.remove_hooks()
    assert torch.equal(hp(x), x)


def test_sparse_autoencoder():
    sae = SparseAutoencoder(d_in=16, d_sae=64)
    x = torch.randn(8, 16)

    f = sae.encode(x)
    assert f.shape == (8, 64)
    assert (f >= 0).all()  # ReLU output non-negative

    x_hat = sae.decode(f)
    assert x_hat.shape == (8, 16)

    x_hat_fwd, f_acts = sae(x)
    assert x_hat_fwd.shape == (8, 16)
    assert f_acts.shape == (8, 64)

    loss = sae.reconstruction_loss(x)
    assert isinstance(float(loss.item()), float)


def test_cross_modal_induction_head_detection():
    attn_stack = [torch.rand(1, 4, 10, 10)]
    image_indices = [0, 1, 2, 3]
    text_indices = [4, 5, 6, 7, 8, 9]

    scores = detect_cross_modal_induction_heads(attn_stack, image_indices, text_indices)
    assert len(scores) == 4
    for key, val in scores.items():
        assert 0.0 <= val <= 1.0


def test_neuron_activation_analysis():
    val = np.array([[1.0, 0.0, 5.0], [0.0, 2.0, 3.0], [4.0, 0.0, 10.0]])
    layer = LayerActivation(layer_name="layer.0", shape=val.shape, values=val)
    run = LayerActivationRun(
        model_family="llava",
        model_name="toy",
        prompt="test prompt",
        layers=[layer],
        tokens=["a", "b", "c"],
    )

    summary = analyze_neuron_activations(run, layer_name="layer.0", neuron_idx=2, top_k=2)
    assert isinstance(summary, NeuronActivationSummary)
    assert summary.layer_name == "layer.0"
    assert summary.neuron_idx == 2
    assert summary.max_activation == 10.0
    assert len(summary.top_tokens) == 2
    assert summary.top_tokens[0] == ("c", 10.0)
