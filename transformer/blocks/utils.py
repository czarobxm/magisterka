"""Module containing utility layers for transformer blocks."""

import torch
from torch import nn
import torch.nn.functional as F


class DownsamplingLayer(nn.Module):
    """
    Downsampling layer for transformer blocks. It reduces the sequence length by grouping
    consecutive elements and linearly transforming them.
    """

    def __init__(self, d_model: int, downsampling_factor: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.downsampling_factor = downsampling_factor
        self.linear = nn.Linear(downsampling_factor * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.size()
        x = x.contiguous().view(
            -1,
            seq_len // self.downsampling_factor,
            self.downsampling_factor * self.d_model,
        )
        return self.linear(x)


class UpsamplingLayer(nn.Module):
    """
    Upsampling layer for transformer blocks. It increases the sequence length by linearly
    transforming consecutive elements.
    """

    def __init__(self, d_model, upsampling_factor) -> None:
        super().__init__()
        self.d_model = d_model
        self.upsampling_factor = upsampling_factor
        self.linear = nn.Linear(d_model, d_model * upsampling_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.size()
        x = self.linear(x)
        return x.reshape(-1, seq_len * self.upsampling_factor, self.d_model)


class ShiftRight(nn.Module):
    """
    Shifts the input tensor to the right by padding zeros. Used only in training.
    """

    def __init__(self, shift: int) -> None:
        super().__init__()
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        seq_len = x.size(1)
        if x.ndim == 2:
            shift_tuple = (self.shift, 0)
        elif x.ndim >= 2:
            shift_tuple = (0, 0, self.shift, 0)
        else:
            raise ValueError("Input tensor can't be 1D.")
        x = F.pad(x, shift_tuple, "constant", 0)
        return x[:, :seq_len]
