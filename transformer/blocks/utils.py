"""Module containing utility layers for transformer blocks."""

import torch
from torch import nn


class DownsamplingLayer(nn.Module):
    """
    Downsampling layer for transformer blocks. It reduces the sequence length by grouping
    consecutive elements and linearly transforming them.
    """

    def __init__(self, d_model: int, downsampling_factor: int) -> None:
        """
        Initialize the downsampling layer.

        Args:
            d_model (int): Dimension of the model
            downsampling_factor (int): Factor by which to reduce the sequence length
        """
        super().__init__()
        self.d_model = d_model
        self.downsampling_factor = downsampling_factor
        self.linear = nn.Linear(downsampling_factor * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the downsampling layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: Downsampled tensor of shape (batch_size, seq_len // downsampling_factor, d_model)
        """
        batch_size, seq_len, _ = x.size()
        new_seq_len = seq_len // self.downsampling_factor
        x = x.contiguous().view(
            batch_size, new_seq_len, self.downsampling_factor * self.d_model
        )
        return self.linear(x)


class UpsamplingLayer(nn.Module):
    """
    Upsampling layer for transformer blocks. It increases the sequence length by linearly
    transforming consecutive elements.
    """

    def __init__(self, d_model: int, upsampling_factor: int) -> None:
        """
        Initialize the upsampling layer.

        Args:
            d_model (int): Dimension of the model
            upsampling_factor (int): Factor by which to increase the sequence length
        """
        super().__init__()
        self.d_model = d_model
        self.upsampling_factor = upsampling_factor
        self.linear = nn.Linear(d_model, d_model * upsampling_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the upsampling layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: Upsampled tensor of shape (batch_size, seq_len * upsampling_factor, d_model)
        """
        batch_size, seq_len, _ = x.size()
        x = self.linear(x)
        return x.contiguous().view(
            batch_size, seq_len * self.upsampling_factor, self.d_model
        )


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
