import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_mean_pool
from typing import Callable, Any

from .gin import GINEncoder
from .mamba_model import MambaBlock
from .bidirectional_mamba import BiMambaBlock, create_bidirectional_mamba_layers
from .mlp_head import MLPHead
from .kan import KANDynamicMixture


class GINMambaHybrid(nn.Module):
    def __init__(
        self,
        node_features: int,
        d_model: int,
        gin_hidden: int = 64,
        gin_layers: int = 3,
        mamba_state: int = 16,
        mamba_conv: int = 4,
        mamba_expand: int = 2,
        mamba_layers: int = 1,
        bidirectional: bool = False,
        mlp_hidden: int = 64,
        mlp_layers: int = 2,
        num_tasks: int = 12,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.gin = GINEncoder(
            in_channels=node_features,
            hidden_channels=gin_hidden,
            num_layers=gin_layers,
            out_channels=d_model,
            dropout=dropout,
        )

        if bidirectional and mamba_layers > 0:
            self.mamba_layers = create_bidirectional_mamba_layers(
                d_model=d_model,
                d_state=mamba_state,
                d_conv=mamba_conv,
                expand=mamba_expand,
                num_layers=mamba_layers,
            )
        else:
            self.mamba_layers = nn.ModuleList(
                [
                    MambaBlock(
                        d_model=d_model,
                        d_state=mamba_state,
                        d_conv=mamba_conv,
                        expand=mamba_expand,
                    )
                    for _ in range(mamba_layers)
                ]
            )

        self.kdm = KANDynamicMixture(d_model)

        self.mlp = MLPHead(
            in_channels=d_model,
            hidden_channels=mlp_hidden,
            out_channels=num_tasks,
            num_layers=mlp_layers,
            dropout=dropout,
        )

    def encode_atoms(self, data: Any, ordering_func: Callable) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, "edge_attr", None)

        h = self.gin(x, edge_index, edge_attr=edge_attr)

        if len(self.mamba_layers) == 0:
            return h

        perm_output = ordering_func(data, descending=False)
        if isinstance(perm_output, tuple):
            perm, scores = perm_output
            h = h * scores.unsqueeze(-1)
        else:
            perm = perm_output

        inv_perm = torch.argsort(perm)

        h_ordered = h[perm]
        batch_perm = batch[perm]
        dense_x, mask = to_dense_batch(h_ordered, batch_perm)

        for mamba_layer in self.mamba_layers:
            dense_x = mamba_layer(dense_x)

        mask_expanded = mask.unsqueeze(-1).expand_as(dense_x)
        h_mamba_ordered = dense_x[mask_expanded].view(-1, dense_x.size(-1))

        h_mamba = h_mamba_ordered[inv_perm]

        h_fused = self.kdm(h, h_mamba)
        return h_fused

    def forward(self, data: Any, ordering_func: Callable) -> torch.Tensor:
        h_fused = self.encode_atoms(data, ordering_func)
        pooled = global_mean_pool(h_fused, data.batch)
        logits = self.mlp(pooled)
        return logits
