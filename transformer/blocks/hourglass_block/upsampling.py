import torch
from torch import nn


class UpsamplingLayer(nn.Module):
    def __init__(self, d_model: int, upsampling_factor: int, upsampling_type: str):
        super().__init__()
        self.d_model = d_model
        self.factor = upsampling_factor
        if upsampling_type == "repeat":
            self.upsampling_layer = RepeatUpsample(
                d_model=d_model, upsampling_factor=upsampling_factor
            )
        elif upsampling_type == "linear":
            self.upsampling_layer = LinearUpsampling(
                d_model=d_model, upsampling_factor=upsampling_factor
            )
        else:
            raise ValueError(f"Invalid upsampling type: {upsampling_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsampling_layer(x)


class LinearUpsampling(nn.Module):
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


class RepeatUpsample(nn.Module):
    def __init__(self, d_model: int, upsampling_factor: int):
        super().__init__()
        self.d_model = d_model
        self.upsampling_factor = upsampling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.repeat_interleave(self.upsampling_factor, dim=-2)
