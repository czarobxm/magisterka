import torch
from torch import nn

from transformer.positional_encoding.base import BasePositionalEncoding


class LearnablePositionalEncoding(BasePositionalEncoding):
    def __init__(self, max_length: int, d_model: int, device="cpu"):
        super().__init__(max_length, d_model)
        self.device = device

        self.encoding = nn.Parameter(torch.rand(max_length, d_model))
        self.to(device)

    def forward(self, x):
        return x + self.encoding[: x.size(1)]
