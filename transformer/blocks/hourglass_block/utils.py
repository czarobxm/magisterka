"""Module containing utility layers for transformer blocks."""

import torch
from torch import nn


class ShiftRight(nn.Module):
    """
    Shifts the input tensor to the right by padding zeros. Used only in training.
    """

    def __init__(self, shift: int) -> None:
        """
        Initialize the shift right layer.

        Args:
            shift (int): Number of positions to shift the input
        """
        super().__init__()
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the shift right layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len) or (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: Shifted tensor of the same shape as input

        Raises:
            ValueError: If input tensor is not 2D or 3D
        """
        if not self.training:
            return x

        if x.ndim not in (2, 3):
            raise ValueError("Input tensor must be 2D or 3D.")

        _, seq_len, *_ = x.size()
        pad_shape = (self.shift, 0) if x.ndim == 2 else (0, 0, self.shift, 0)
        return nn.functional.pad(x, pad_shape)[:, :seq_len]
