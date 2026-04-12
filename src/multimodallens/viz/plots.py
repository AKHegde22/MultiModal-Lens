"""Plotting helpers for Gradio outputs."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_alignment(tokens: list[str], matrix: np.ndarray):
    """Create token x patch heatmap plot."""
    fig, ax = plt.subplots(figsize=(10, max(3, len(tokens) * 0.22)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(tokens)))
    ax.set_yticklabels(tokens)
    ax.set_xlabel("Image Patches")
    ax.set_ylabel("Text Tokens")
    ax.set_title("Token-Image Alignment")
    fig.colorbar(im, ax=ax, shrink=0.75)
    fig.tight_layout()
    return fig


def plot_token_scores(tokens: list[str], scores: np.ndarray):
    """Create bar plot of token contribution scores."""
    fig, ax = plt.subplots(figsize=(10, max(3, len(tokens) * 0.22)))
    y = np.arange(len(tokens))
    ax.barh(y, scores, color="#1f77b4")
    ax.set_yticks(y)
    ax.set_yticklabels(tokens)
    ax.invert_yaxis()
    ax.set_xlabel("Contribution Score")
    ax.set_title("Per-token Cross-modal Similarity")
    fig.tight_layout()
    return fig


def plot_faithfulness_curves(deletion: list[float], insertion: list[float]):
    """Plot deletion/insertion curves."""
    x = np.linspace(0.0, 1.0, len(deletion))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, deletion, label="Deletion Drop", color="#d62728")
    ax.plot(x, insertion, label="Insertion Gain", color="#2ca02c")
    ax.set_xlabel("Masked Fraction")
    ax.set_ylabel("Score Change")
    ax.set_title("Faithfulness Perturbation Curves")
    ax.legend()
    fig.tight_layout()
    return fig
