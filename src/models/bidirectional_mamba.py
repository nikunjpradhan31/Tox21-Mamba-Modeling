import torch
import torch.nn as nn
from .mamba_model import MambaBlock


class BiMambaBlock(nn.Module):
    """
    Bidirectional Mamba block that processes sequences in both forward and backward directions.
    """
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.forward_mamba = MambaBlock(d_model, d_state, d_conv, expand)
        self.backward_mamba = MambaBlock(d_model, d_state, d_conv, expand)
        self.mixer = nn.Linear(d_model * 2, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Output tensor of same shape
        """
        # Forward pass
        forward_out = self.forward_mamba(x)
        
        # Backward pass (reverse sequence, process, then reverse back)
        backward_out = self.backward_mamba(torch.flip(x, dims=[1]))
        backward_out = torch.flip(backward_out, dims=[1])
        
        # Combine both directions
        combined = torch.cat([forward_out, backward_out], dim=-1)
        return self.mixer(combined)


def create_bidirectional_mamba_layers(d_model: int, d_state: int, d_conv: int, 
                                     expand: int, num_layers: int) -> nn.ModuleList:
    """
    Create a module list of bidirectional Mamba layers.
    
    Args:
        d_model: Model dimension
        d_state: State dimension
        d_conv: Convolution dimension  
        expand: Expansion factor
        num_layers: Number of bidirectional layers
    
    Returns:
        ModuleList of BiMambaBlock layers
    """
    return nn.ModuleList([
        BiMambaBlock(d_model, d_state, d_conv, expand)
        for _ in range(num_layers)
    ])