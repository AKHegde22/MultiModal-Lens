"""Sparse Autoencoder (SAE) integration for dictionary learning over activations."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder (SAE) for decomposing activation vectors into interpretable features.

    Architecture:
      Encoder: f = ReLU((x - b_dec) @ W_enc + b_enc)
      Decoder: x_hat = f @ W_dec + b_dec
    """

    def __init__(self, d_in: int, d_sae: int) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae

        self.W_enc = nn.Parameter(torch.randn(d_in, d_sae) * (1.0 / (d_in ** 0.5)))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_in) * (1.0 / (d_sae ** 0.5)))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activation tensor x into sparse feature activations f."""
        x_centered = x - self.b_dec
        return F.relu(x_centered @ self.W_enc + self.b_enc)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """Decode feature activations f back into activation space x_hat."""
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (x_hat, feature_activations)."""
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f

    def feature_acts(self, x: torch.Tensor) -> torch.Tensor:
        """Return sparse feature activations for input x."""
        return self.encode(x)

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute relative L2 reconstruction loss ||x - x_hat||^2 / ||x||^2."""
        x_hat, _ = self.forward(x)
        l2_diff = (x - x_hat).pow(2).sum(dim=-1)
        l2_orig = x.pow(2).sum(dim=-1).clamp(min=1e-8)
        return (l2_diff / l2_orig).mean()
