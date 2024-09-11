from typing import Tuple
import torch
from torch import nn


class BaseAttentionMechanism(nn.Module):
    """
    Base class for attention mechanisms.

    This class defines the basic structure and interface that all attention
    mechanism implementations should follow.
    """

    def __init__(self, d_model: int, num_heads: int) -> None:
        """
        Initialize the BaseAttentionMechanism.

        Args:
            d_model (int): The dimension of the model.
            num_heads (int): The number of attention heads.

        Raises:
            ValueError: If d_model is not divisible by num_heads.
        """
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_head = d_model // num_heads

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool = False,
        inference: bool = False,
    ) -> torch.Tensor:
        """
        Perform the attention mechanism computation.

        Args:
            query (torch.Tensor): The query tensor.
            key (torch.Tensor): The key tensor.
            value (torch.Tensor): The value tensor.
            causal (bool, optional): Whether to use causal attention. Defaults to False.
            inference (bool, optional): Whether to use inference mode. Defaults to False.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("forward method must be implemented by subclasses")

    def multihead_reshape(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reshape input tensors for multi-head attention.

        Args:
            query (torch.Tensor): The query tensor.
            key (torch.Tensor): The key tensor.
            value (torch.Tensor): The value tensor.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError(
            "multihead_reshape method must be implemented by subclasses"
        )

    def inference(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform inference using the attention mechanism.

        Args:
            query (torch.Tensor): The query tensor.
            key (torch.Tensor): The key tensor.
            value (torch.Tensor): The value tensor.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("inference method must be implemented by subclasses")
