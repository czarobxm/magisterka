"""
Encoder module of the transformer model.
"""

from typing import Union

import torch
from torch import nn

from transformer.layers.multi_head_attention.attention_mechanism.attn_params import (
    LinearAttnParams,
    VanillaParams,
    PerformerParams,
    CosformerParams,
)
from transformer.layers.multi_head_attention import MultiHeadAttention
from transformer.layers import FeedForward
from transformer.blocks.utils import ShiftRight


class BlockLayer(nn.Module):
    """Single block layer with multi-head attention and feed forward network."""

    def __init__(
        self,
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
        has_outproj: bool = True,
        act_fun: nn.Module = None,
        norm_before: bool = True,
        device: str = "cpu",
    ) -> None:
        """
        Init single encoder layer with multi-head attention and feed forward network
        connected with residual connection and layer normalization.

        :param mha_params: multi-head attention parameters
        :param n_layers: number of layers
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_head = d_model // num_heads
        self.method_params = method_params
        self.apply_rotary_pos_enc = apply_rotary_pos_enc
        self.dropout = dropout
        self.has_outproj = has_outproj
        self.act_fun = act_fun
        self.norm_before = norm_before
        self.device = device

        self.attention = MultiHeadAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            method_params=self.method_params,
            apply_rotary_pos_enc=self.apply_rotary_pos_enc,
            dropout=self.dropout,
            has_outproj=self.has_outproj,
            act_fun=self.act_fun,
            device=device,
        )
        self.norm1 = nn.LayerNorm(self.d_model)
        self.dropout1 = nn.Dropout(p=self.dropout)

        self.ffn = FeedForward(
            d_model=self.d_model,
            hidden=2 * self.d_model,
            drop_prob=self.dropout,
        )
        self.norm2 = nn.LayerNorm(self.d_model)
        self.dropout2 = nn.Dropout(p=self.dropout)

        self.alpha = nn.Parameter(torch.tensor([1]), requires_grad=False)

        self.init_weights()

        self.to(device)

    def init_weights(self) -> None:
        """Initialize weights of the model."""
        self.attention.init_weights()
        self.ffn.init_weights()
        # torch.nn.init.xavier_uniform_(self.norm1.weight)
        # torch.nn.init.xavier_uniform_(self.norm2.weight)
        # torch.nn.init.xavier_uniform_(self.norm1.bias)
        # torch.nn.init.xavier_uniform_(self.norm2.bias)

    def forward(
        self,
        x: torch.Tensor,
        key_value: torch.Tensor = None,
        causal: bool = False,
        inference: bool = False,
    ) -> torch.Tensor:
        """Produces the output of the single encoder layer."""
        residual = x
        # PreNorm
        if self.norm_before:
            x = self.norm1(x)
            key_value = self.norm1(key_value) if key_value is not None else None
        # Self Attention
        if key_value is not None:
            x = self.attention(
                query=x,
                key=key_value,
                value=key_value,
                causal=causal,
                inference=inference,
            )
        else:
            x = self.attention(
                query=x, key=x, value=x, causal=causal, inference=inference
            )
        # Add
        x = self.dropout1(x) + residual
        # PostNorm
        if not self.norm_before:
            x = self.norm1(x)

        residual = x
        # PreNorm
        if self.norm_before:
            x = self.norm2(x)
        # Feed Forward
        x = self.ffn(x)
        # Add
        x = self.dropout2(x) + residual
        # PostNorm
        if not self.norm_before:
            x = self.norm2(x)

        return x


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
        has_outproj: bool = True,
        act_fun: str = None,
        norm_before: bool = True,
        device: str = "cpu",
    ) -> None:
        """Init encoder - stack :param n_layers: the encoder layers on each other"""
        super().__init__()
        self.d_model = d_model
        if isinstance(n_layers, int):
            self.n_layers = n_layers
        elif isinstance(n_layers, list) and len(n_layers) == 1:
            self.n_layers = n_layers[0]
        else:
            raise ValueError(
                "n_layers must be an integer or a list with a single integer"
            )
        self.num_heads = num_heads
        self.method_params = method_params
        self.apply_rotary_pos_enc = apply_rotary_pos_enc
        self.dropout = dropout
        self.has_outproj = has_outproj
        self.act_fun = act_fun
        self.norm_before = norm_before
        self.device = device

        self.shift_right = ShiftRight(shift=1)

        self.layers = nn.ModuleList(
            [
                BlockLayer(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    method_params=self.method_params,
                    apply_rotary_pos_enc=self.apply_rotary_pos_enc,
                    dropout=self.dropout,
                    has_outproj=self.has_outproj,
                    act_fun=self.act_fun,
                    norm_before=self.norm_before,
                    device=self.device,
                )
                for _ in range(self.n_layers)
            ]
        )

        if self.has_outproj:
            self.out_proj = nn.Linear(self.d_model, self.d_model)

        self.to(device)

    def forward(
        self,
        x: torch.Tensor,
        key_value: torch.Tensor = None,
        causal: bool = False,
        inference: bool = False,
    ) -> torch.Tensor:
        """Produces the output of the encoder block."""
        for layer in self.layers:
            x = layer(x=x, key_value=key_value, causal=causal, inference=inference)
            if key_value is not None:
                key_value = None
        if self.has_outproj:
            return self.out_proj(x)
        return x
