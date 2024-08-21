import math

import torch

from transformer.positional_encoding.base import BasePositionalEncoding


class SinusoidalPositionalEncoding(BasePositionalEncoding):
    def __init__(self, max_length, d_model, device="mps"):
        super().__init__(max_length, d_model)
        pe = torch.zeros(self.max_length, d_model)
        position = torch.arange(0, self.max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

        self.to(device)

    def forward(self, x):
        return x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
