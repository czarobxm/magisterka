"""
Encoder module of the transformer model.
"""

from typing import Union, List

import torch
from torch import nn

from transformer.multi_head_attention.attention_mechanism.attn_params import (
    LinearAttnParams,
    VanillaParams,
    PerformerParams,
    CosformerParams,
)

from transformer.blocks.single_layer import TransformerLayer


class Block(nn.Module):
    """
    Single block in transformer model. Stack of layers with multi-head attention
    and feed forward network.
    """

    def __init__(
        self,
        n_layers: Union[int],
        d_model: int,
        num_heads: int,
        method_params: Union[
            LinearAttnParams,
            VanillaParams,
            PerformerParams,
            CosformerParams,
        ],
        apply_rotary_pos_enc: bool = True,
        dropout: float = 0.1,
        act_fun: str = None,
        post_norm: bool = False,
        device: str = "cpu",
    ) -> None:
        """Init encoder - stack :param n_layers: the encoder layers on each other"""
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device

        self.n_layers = self._validate_n_layers(n_layers)

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    method_params=method_params,
                    apply_rotary_pos_enc=apply_rotary_pos_enc,
                    dropout=dropout,
                    act_fun=act_fun,
                    post_norm=post_norm,
                    device=self.device,
                )
                for _ in range(self.n_layers)
            ]
        )

        self.to(device)

    def _validate_n_layers(self, n_layers: Union[int, List[int]]) -> int:
        """Validate and return the number of layers."""
        if isinstance(n_layers, int):
            return n_layers
        elif isinstance(n_layers, list) and len(n_layers) == 1:
            return n_layers[0]
        else:
            raise ValueError(
                "n_layers must be an integer or a list with a single integer"
            )

    def forward(
        self,
        x: torch.Tensor,
        key_value: torch.Tensor = None,
        causal: bool = True,
        inference: bool = False,
    ) -> torch.Tensor:
        """Produces the output of the encoder block."""
        for layer in self.layers:
            x = layer(x=x, key_value=key_value, causal=causal, inference=inference)
            key_value = None  # Set key_value to None after first layer
        return x
