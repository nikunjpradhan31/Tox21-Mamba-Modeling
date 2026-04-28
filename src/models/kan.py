import torch
import torch.nn as nn
import torch.nn.functional as F

class KANLayer(nn.Module):
    """
    A simplified Kolmogorov-Arnold Network (KAN) Layer approximation.
    Uses SiLU activations and a non-linear combination to approximate univariate functions.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Base linear transformation
        self.base_linear = nn.Linear(in_dim, out_dim)
        
        # Non-linear univariate function approximator (using a hidden expansion)
        self.spline_approx = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.SiLU(),
            nn.Linear(in_dim * 2, out_dim)
        )
        
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        # KAN computes: sum of univariate functions
        base = self.base_linear(x)
        spline = self.spline_approx(x)
        return self.layer_norm(base + spline)


class KANDynamicMixture(nn.Module):
    """
    Fuses local (GNN) and global (Mamba) features using a KAN.
    """
    def __init__(self, d_model: int):
        super().__init__()
        # Takes [local, global] concatenated -> d_model
        self.kan = KANLayer(d_model * 2, d_model)
        
    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        # Concatenate features along feature dimension
        x = torch.cat([local_feat, global_feat], dim=-1)
        # Non-linear KAN fusion
        return self.kan(x)
