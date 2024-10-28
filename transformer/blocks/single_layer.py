"""
Encoder module of the transformer model.
"""

from typing import Union, Optional

import torch
from torch import nn

from transformer.multi_head_attention.attention_mechanism.attn_params import (
    LinearAttnParams,
    VanillaParams,
    PerformerParams,
    CosformerParams,
)
from transformer.multi_head_attention import MultiHeadAttention
from transformer.feed_forward import FeedForward


class TransformerLayer(nn.Module):
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
        post_norm: bool = False,
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
        self.post_norm = post_norm
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
            hidden=4 * self.d_model,
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
        if self.post_norm:
            x = self.norm1(x + self._attention(x, key_value, causal, inference))
        else:
            x = x + self._attention(
                self.norm1(x),
                self.norm1(key_value) if key_value is not None else None,
                causal,
                inference,
            )
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
                kv,
                kv,
                causal=causal,
                inference=inference,
            )
        )

    def _feedforward_block(self, x: torch.Tensor) -> torch.Tensor:
        if self.post_norm:
            x = self.norm2(x + self.dropout2(self.ffn(x)))
        else:
            x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x
