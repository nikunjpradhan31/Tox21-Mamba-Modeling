import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv


class GINEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: int,
        dropout: float = 0.0,
        edge_dim: int = 9,
    ):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        # Encode node and edge features to hidden_channels
        self.atom_encoder = nn.Linear(in_channels, hidden_channels)
        self.bond_encoder = nn.Linear(edge_dim, hidden_channels)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GINEConv(
                    nn.Sequential(
                        nn.Linear(hidden_channels, hidden_channels),
                        nn.ReLU(),
                        nn.Linear(hidden_channels, hidden_channels),
                    )
                )
            )

        # Jumping Knowledge (JK): concatenate all layer outputs and project
        self.jk_linear = nn.Linear(hidden_channels * (num_layers + 1), out_channels)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None
    ) -> torch.Tensor:
        
        # Initial encoding
        x = self.atom_encoder(x)
        
        if edge_attr is not None:
            edge_attr = self.bond_encoder(edge_attr)
        else:
            # Fallback if no edge attributes provided (should not happen with Tox21 dataset)
            edge_attr = torch.zeros((edge_index.size(1), x.size(1)), device=x.device)

        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = torch.relu(x)
            if self.dropout > 0:
                x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)

        # Jumping Knowledge: concatenate multi-scale representations
        jk_x = torch.cat(xs, dim=-1)
        out = self.jk_linear(jk_x)

        return out
