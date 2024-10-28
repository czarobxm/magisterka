import torch
from torch import nn


class DownsamplingLayer(nn.Module):
    def __init__(self, d_model: int, downsampling_factor: int, downsampling_type: str):
        super().__init__()
        self.d_model = d_model
        self.factor = downsampling_factor
        if downsampling_type == "avg":
            self.downsampling_layer = AvgPool(
                d_model=d_model, downsampling_factor=downsampling_factor
            )
        elif downsampling_type == "linear":
            self.downsampling_layer = LinearPool(
                d_model=d_model, downsampling_factor=downsampling_factor
            )
        else:
            raise ValueError(f"Invalid downsampling type: {downsampling_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.downsampling_layer(x)


class AvgPool(nn.Module):
    def __init__(self, d_model, downsampling_factor: int):
        super().__init__()
        self.d_model = d_model
        self.downsampling_factor = downsampling_factor
        self.avg_pool = nn.AvgPool1d(
            kernel_size=downsampling_factor, stride=downsampling_factor
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avg_pool(x.transpose(-1, -2)).transpose(-2, -1)


class LinearPool(nn.Module):
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
