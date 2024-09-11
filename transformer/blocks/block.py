"""
Encoder module of the transformer model.
"""

from typing import Union, Optional, List

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
        act_fun: Optional[nn.Module] = None,
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
        self.norm_before = norm_before
        self.device = device

        self.attention = MultiHeadAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            method_params=method_params,
            apply_rotary_pos_enc=apply_rotary_pos_enc,
            dropout=dropout,
            has_outproj=has_outproj,
            act_fun=act_fun,
            device=device,
        )

        self.norm1 = nn.LayerNorm(self.d_model)
        self.dropout1 = nn.Dropout(p=dropout)

        self.ffn = FeedForward(
            d_model=self.d_model,
            hidden=2 * self.d_model,
            drop_prob=dropout,
        )

        self.norm2 = nn.LayerNorm(self.d_model)
        self.dropout2 = nn.Dropout(p=dropout)

        self.to(device)

    def forward(
        self,
        x: torch.Tensor,
        key_value: torch.Tensor = None,
        causal: bool = False,
        inference: bool = False,
    ) -> torch.Tensor:
        """Produces the output of the single encoder layer."""
        x = self._attention_block(x, key_value, causal, inference)
        x = self._feedforward_block(x)
        return x

    def _attention_block(
        self,
        x: torch.Tensor,
        key_value: Optional[torch.Tensor],
        causal: bool,
        inference: bool,
    ) -> torch.Tensor:
        if self.norm_before:
            x = x + self._attention(
                self.norm1(x),
                self.norm1(key_value) if key_value is not None else None,
                causal,
                inference,
            )
        else:
            x = self.norm1(x + self._attention(x, key_value, causal, inference))
        return x

    def _attention(
        self,
        x: torch.Tensor,
        key_value: Optional[torch.Tensor],
        causal: bool,
        inference: bool,
    ) -> torch.Tensor:
        kv = key_value if key_value is not None else x
        return self.dropout1(
            self.attention(
                x,
                x,
                x,
                causal=causal,
                inference=inference,
            )
        )

    def _feedforward_block(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_before:
            x = x + self.dropout2(self.ffn(self.norm2(x)))
        else:
            x = self.norm2(x + self.dropout2(self.ffn(x)))
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
        self.num_heads = num_heads
        self.has_outproj = has_outproj
        self.device = device

        self.n_layers = self._validate_n_layers(n_layers)

        self.layers = nn.ModuleList(
            [
                BlockLayer(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    method_params=method_params,
                    apply_rotary_pos_enc=apply_rotary_pos_enc,
                    dropout=dropout,
                    has_outproj=self.has_outproj,
                    act_fun=act_fun,
                    norm_before=norm_before,
                    device=self.device,
                )
                for _ in range(self.n_layers)
            ]
        )

        if self.has_outproj:
            self.out_proj = nn.Linear(self.d_model, self.d_model)

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
        causal: bool = False,
        inference: bool = False,
    ) -> torch.Tensor:
        """Produces the output of the encoder block."""
        for layer in self.layers:
            x = layer(x=x, key_value=key_value, causal=causal, inference=inference)
            key_value = None  # Set key_value to None after first layer

        if self.has_outproj:
            return self.out_proj(x)
        return x
