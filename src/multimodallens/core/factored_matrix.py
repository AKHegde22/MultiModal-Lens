"""FactoredMatrix: Efficient representation of low-rank weight matrices for circuit analysis."""

from __future__ import annotations
import torch

class FactoredMatrix:
    """Represents a matrix M = A @ B without computing the full product.
    
    Used for analyzing OV circuits (W_V @ W_O) and QK circuits (W_Q @ W_K^T)
    in transformer attention heads without materializing the full (d_model x d_model) matrix.
    
    Key operations (SVD, norms, eigenvalues) are computed efficiently using
    the factored representation, bounded by O(d_head^2 * d_model) instead of O(d_model^3).
    """
    
    def __init__(self, A: torch.Tensor, B: torch.Tensor):
        """Initialize with factor matrices A and B."""
        if A.shape[-1] != B.shape[-2]:
            raise ValueError(f"Incompatible shapes for A {A.shape} and B {B.shape}")
        
        # Determine common batch dimensions, broadcast if necessary
        a_batch = A.shape[:-2]
        b_batch = B.shape[:-2]
        
        self.batch_shape = torch.broadcast_shapes(a_batch, b_batch)
        
        # Expand A and B to have the matching batch shape
        self.A = A.expand(self.batch_shape + A.shape[-2:])
        self.B = B.expand(self.batch_shape + B.shape[-2:])

    @property
    def shape(self) -> tuple[int, ...]:
        return self.batch_shape + (self.A.shape[-2], self.B.shape[-1])
    
    @property 
    def device(self) -> torch.device:
        return self.A.device
    
    @property
    def dtype(self) -> torch.dtype:
        return self.A.dtype
    
    def materialized(self) -> torch.Tensor:
        """Compute and return the full A @ B product. Use sparingly."""
        return self.A @ self.B
        
    def svd(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Efficient SVD without materializing A @ B.
        Uses QR decomposition on A and B independently.
        """
        # A: (..., m, k), B: (..., k, n)
        # Q_A, R_A = qr(A) where Q_A: (..., m, k), R_A: (..., k, k)
        Q_A, R_A = torch.linalg.qr(self.A)
        
        # Q_B, R_B = qr(B^T) where Q_B: (..., n, k), R_B: (..., k, k)
        Q_B, R_B = torch.linalg.qr(self.B.transpose(-1, -2))
        
        # Mid matrix: (..., k, k)
        mid = R_A @ R_B.transpose(-1, -2)
        
        # SVD of mid matrix
        U_mid, S, Vh_mid = torch.linalg.svd(mid)
        
        # U = Q_A @ U_mid
        U = Q_A @ U_mid
        # Vh = Vh_mid @ Q_B^T
        Vh = Vh_mid @ Q_B.transpose(-1, -2)
        
        return U, S, Vh
        
    def eigenvalues(self) -> torch.Tensor:
        """Compute eigenvalues efficiently via B @ A (k x k instead of m x m)."""
        if self.shape[-2] != self.shape[-1]:
            raise ValueError("Eigenvalues are only defined for square matrices (m == n).")
            
        # Non-zero eigenvalues of AB are the same as BA
        BA = self.B @ self.A
        return torch.linalg.eigvals(BA)
        
    def norm(self, kind: str = 'fro') -> torch.Tensor:
        """Frobenius norm via trace(B^T @ A^T @ A @ B) = trace((A^T A)(B B^T))."""
        if kind == 'fro':
            # trace( (A^T A) @ (B B^T) )
            AtA = self.A.transpose(-1, -2) @ self.A
            BBt = self.B @ self.B.transpose(-1, -2)
            prod = AtA @ BBt
            return torch.diagonal(prod, dim1=-2, dim2=-1).sum(dim=-1).sqrt()
        elif kind == 'nuc':
            _, S, _ = self.svd()
            return S.sum(dim=-1)
        else:
            return torch.linalg.matrix_norm(self.materialized(), ord=kind)
        
    def trace(self) -> torch.Tensor:
        """Trace via Tr(AB) = sum_ij A_ij * B_ji."""
        if self.shape[-2] != self.shape[-1]:
            raise ValueError("Trace is only defined for square matrices (m == n).")
        
        BA = self.B @ self.A
        return torch.diagonal(BA, dim1=-2, dim2=-1).sum(dim=-1)
    
    def __matmul__(self, other: torch.Tensor) -> torch.Tensor:
        """Right multiply: (A @ B) @ x = A @ (B @ x)."""
        return self.A @ (self.B @ other)
        
    def __rmatmul__(self, other: torch.Tensor) -> torch.Tensor:
        """Left multiply: x @ (A @ B) = (x @ A) @ B."""
        return (other @ self.A) @ self.B
        
    def compose(self, other: FactoredMatrix) -> FactoredMatrix:
        """Compose two factored matrices: (A1 @ B1) @ (A2 @ B2) = A1 @ (B1 @ A2) @ B2."""
        mid = self.B @ other.A
        return FactoredMatrix(self.A, mid @ other.B)
        
    @staticmethod
    def ov_circuit(W_V: torch.Tensor, W_O: torch.Tensor) -> FactoredMatrix:
        """Create OV circuit factored matrix for an attention head.
        W_V: (d_model, d_head), W_O: (d_head, d_model)
        OV = W_V @ W_O describes what information the head copies.
        """
        return FactoredMatrix(W_V, W_O)
        
    @staticmethod
    def qk_circuit(W_Q: torch.Tensor, W_K: torch.Tensor) -> FactoredMatrix:
        """Create QK circuit factored matrix for an attention head.
        W_Q: (d_model, d_head), W_K: (d_model, d_head)
        QK = W_Q @ W_K^T describes what positions the head attends to.
        """
        return FactoredMatrix(W_Q, W_K.transpose(-1, -2))
    
    def __repr__(self) -> str:
        return f"FactoredMatrix(shape={self.shape}, rank<={self.A.shape[-1]}, dtype={self.dtype}, device={self.device})"
