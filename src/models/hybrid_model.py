import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_mean_pool
from typing import Callable, Any

from .gin import GINEncoder
from .mamba_model import MambaBlock
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
        mlp_hidden: int = 64,
        mlp_layers: int = 2,
        num_tasks: int = 12,
        dropout: float = 0.0,
    ):
        super().__init__()

        # GIN encoder for local graph structure
        self.gin = GINEncoder(
            in_channels=node_features,
            hidden_channels=gin_hidden,
            num_layers=gin_layers,
            out_channels=d_model,
            dropout=dropout,
        )

        # Mamba layers for sequence modeling
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

        # KAN Dynamic Mixture to fuse local (GNN) and global (Mamba) features
        self.kdm = KANDynamicMixture(d_model)

        # MLP head for task predictions (d_model + 1024 for Morgan Fingerprints)
        self.mlp = MLPHead(
            in_channels=d_model + 1024,
            hidden_channels=mlp_hidden,
            out_channels=num_tasks,
            num_layers=mlp_layers,
            dropout=dropout,
        )

    def forward(self, data: Any, ordering_func: Callable) -> torch.Tensor:
        """
        data: PyG Batch object
        ordering_func: Callable that takes 'data' and returns a node permutation tensor
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, 'edge_attr', None)

        # 1. Local graph encoding (GIN + Edge features + JK)
        h = self.gin(x, edge_index, edge_attr=edge_attr)
        
        # Capture strictly local pooled features BEFORE serialization
        pooled_local = global_mean_pool(h, batch)

        # 2. Reordering
        perm_output = ordering_func(data)
        if isinstance(perm_output, tuple):
            perm, scores = perm_output
            # Soft gating to allow gradient flow to learned ordering module
            h = h * scores.unsqueeze(-1)
        else:
            perm = perm_output

        h = h[perm]
        batch_perm = batch[perm]

        # 3. Form sequences (to dense batch)
        dense_x, mask = to_dense_batch(h, batch_perm)

        # 4. Mamba sequence modeling
        for mamba_layer in self.mamba_layers:
            dense_x = mamba_layer(dense_x)

        # 5. Masked mean pooling over the sequence (global features)
        # mask shape: (batch_size, max_seq_len)
        mask_float = mask.float().unsqueeze(-1)  # (batch_size, max_seq_len, 1)
        pooled_global = (dense_x * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(
            min=1e-9
        )

        # 6. KAN Dynamic Mixture (Fusion)
        fused = self.kdm(pooled_local, pooled_global)

        # 7. Knowledge Graph / Global Topological Injection (Morgan Fingerprints)
        if hasattr(data, 'fp') and data.fp is not None:
            fp = data.fp
            # PyG batching stacked them as (batch_size, 1, 1024). We need (batch_size, 1024)
            if fp.dim() == 3 and fp.size(1) == 1:
                fp = fp.squeeze(1)
            elif fp.dim() == 1:
                fp = fp.unsqueeze(0)
            fused = torch.cat([fused, fp], dim=-1)
        else:
            fp_dummy = torch.zeros(fused.size(0), 1024, device=fused.device)
            fused = torch.cat([fused, fp_dummy], dim=-1)

        # 8. Classification
        logits = self.mlp(fused)
        return logits
