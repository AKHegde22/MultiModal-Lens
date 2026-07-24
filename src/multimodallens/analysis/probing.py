"""Linear probing infrastructure for evaluating feature decodability in activation caches."""

from __future__ import annotations

from typing import Any
import numpy as np
from multimodallens.core.activation_cache import ActivationCache


class LinearProbe:
    """Linear classifier probe trained on layer hidden state representations."""

    def __init__(self, layer_name: str, C: float = 1.0) -> None:
        self.layer_name = layer_name
        self.C = C
        self.model: Any | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearProbe:
        """Fit a linear probe on features X and labels y."""
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        while X_arr.ndim > 2:
            X_arr = X_arr.mean(axis=1)

        try:
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(C=self.C, max_iter=1000)
            self.model.fit(X_arr, y_arr)
        except ImportError:
            d_in = X_arr.shape[1]
            unique_y = np.unique(y_arr)
            label0 = unique_y[0]
            label1 = unique_y[1] if len(unique_y) > 1 else unique_y[0]
            y_binary = np.where(y_arr == label0, -1.0, 1.0)
            weights, _, _, _ = np.linalg.lstsq(X_arr, y_binary, rcond=None)
            self.model = ("least_squares_binary", weights, label0, label1)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for features X."""
        X_arr = np.asarray(X)
        while X_arr.ndim > 2:
            X_arr = X_arr.mean(axis=1)

        if hasattr(self.model, "predict"):
            return self.model.predict(X_arr)

        if isinstance(self.model, tuple) and self.model[0] == "least_squares_binary":
            _, weights, label0, label1 = self.model
            scores = X_arr @ weights
            return np.where(scores < 0, label0, label1)

        return np.zeros(X_arr.shape[0], dtype=int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy score on (X, y)."""
        preds = self.predict(X)
        y_arr = np.asarray(y)
        if y_arr.size == 0:
            return 0.0
        return float(np.mean(preds == y_arr))


def evaluate_layer_probes(
    caches: list[ActivationCache],
    labels: list[int] | np.ndarray,
    layer_names: list[str] | None = None,
) -> dict[str, float]:
    """Train and evaluate linear probes across all layers in activation caches.
    
    Returns a dictionary mapping layer_name -> probe accuracy score.
    """
    if not caches:
        return {}

    y = np.asarray(labels)
    if layer_names is None:
        layer_names = list(caches[0].keys())

    scores: dict[str, float] = {}

    for name in layer_names:
        feats = []
        valid_indices = []
        for i, cache in enumerate(caches):
            if name in cache:
                tensor = cache[name]
                arr = tensor.cpu().numpy() if hasattr(tensor, "cpu") else np.asarray(tensor)
                feats.append(arr)
                valid_indices.append(i)

        if not feats:
            continue

        X = np.concatenate([f if f.ndim > 1 else f[np.newaxis, :] for f in feats], axis=0)
        y_sub = y[valid_indices]

        if len(np.unique(y_sub)) < 2:
            scores[name] = 1.0
            continue

        probe = LinearProbe(layer_name=name)
        probe.fit(X, y_sub)
        scores[name] = probe.score(X, y_sub)

    return scores
