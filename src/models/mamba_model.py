import torch
import torch.nn as nn


#import os
# import sys

# mamba2_path = os.path.join(os.path.dirname(__file__), "mamba2-minimal")
# if mamba2_path not in sys.path:
#     sys.path.append(mamba2_path)

from .mamba2_minimal.mamba2 import Mamba2, Mamba2Config


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        chunk_size: int = 64,
    ):
        super().__init__()

        # Adjust headdim if d_inner is not divisible by headdim
        d_inner = expand * d_model
        if d_inner % headdim != 0:
            for h in range(min(64, d_inner), 0, -1):
                if d_inner % h == 0:
                    headdim = h
                    break

        config = Mamba2Config(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            chunk_size=chunk_size,
        )
        self.mamba = Mamba2(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)
        Returns:
        output: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        chunk_size = self.mamba.args.chunk_size

        pad_len = (chunk_size - (seq_len % chunk_size)) % chunk_size
        if pad_len > 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_len))

        y, _ = self.mamba(x)

        if pad_len > 0:
            y = y[:, :seq_len, :]

        return y
