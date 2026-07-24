"""Neuron-level activation diagnostics and top-activating feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from multimodallens.types import LayerActivationRun


@dataclass(slots=True)
class NeuronActivationSummary:
    """Summary statistics for an individual neuron across sequence positions."""

    layer_name: str
    neuron_idx: int
    mean_activation: float
    max_activation: float
    sparsity_l0: float
    top_tokens: list[tuple[str, float]]


def analyze_neuron_activations(
    activations_run: LayerActivationRun,
    layer_name: str,
    neuron_idx: int,
    top_k: int = 5,
) -> NeuronActivationSummary:
    """Analyze activation behavior of a specific neuron across sequence positions."""
    layer = next((l for l in activations_run.layers if l.layer_name == layer_name), None)
    if layer is None:
        raise ValueError(f"Layer '{layer_name}' not found in activation run.")

    val = np.asarray(layer.values)
    if val.ndim > 2:
        val = val[0]

    if neuron_idx >= val.shape[-1]:
        raise ValueError(f"Neuron index {neuron_idx} exceeds layer dimension {val.shape[-1]}.")

    neuron_acts = val[:, neuron_idx]
    seq_len = int(neuron_acts.shape[0])

    mean_act = float(np.mean(neuron_acts))
    max_act = float(np.max(neuron_acts))
    sparsity = float(np.count_nonzero(neuron_acts > 0) / max(seq_len, 1))

    tokens = activations_run.tokens or [f"pos_{i}" for i in range(seq_len)]
    tokens_padded = tokens[:seq_len] + [f"pos_{i}" for i in range(len(tokens), seq_len)]

    top_indices = np.argsort(-neuron_acts)[:top_k]
    top_tokens = [(tokens_padded[idx], float(neuron_acts[idx])) for idx in top_indices]

    return NeuronActivationSummary(
        layer_name=layer_name,
        neuron_idx=neuron_idx,
        mean_activation=mean_act,
        max_activation=max_act,
        sparsity_l0=sparsity,
        top_tokens=top_tokens,
    )
