"""
Multi Head Attention with different attention mechanisms. On top of that, there is
a possibility to add Linformer's projection before the attention mechanism.
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
from transformer.layers.multi_head_attention.attention_mechanism import AttentionMechanism


class MultiHeadAttention(nn.Module):
    """
    Multi Head Attention mechanism.

    :param d_model: dimension of embedding, size of one vector in a sequence
    :param num_heads: number of heads
    :param dropout: dropout rate
    :param method_params: dictionary with method parameters
    :param linear_projection_dim: dimension of the linear projection
    :param rotary_pos_enc: flag determining whether to use rotary position encoding
    :param causal: flag determining whether to use causal attention
    :param has_outproj: flag determining whether to use output projection
    :param act_fun: activation function
    :param device: device
    """

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
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_head = self.d_model // self.num_heads
        self.method_params = method_params
        self.dropout = dropout
        self.has_outproj = has_outproj
        self.apply_rotary_pos_enc = apply_rotary_pos_enc
        self.act_fun = act_fun
        self.device = device

        # Linear transformations
        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)
        if self.has_outproj:
            self.w_o = nn.Linear(self.d_model, self.d_model)

        self.attention_mechanism = AttentionMechanism(
            d_model=self.d_model,
            num_heads=self.num_heads,
            method_params=self.method_params,
            apply_rotary_pos_enc=self.apply_rotary_pos_enc,
            device=self.device,
        )

        self.dropout = nn.Dropout(p=self.dropout)

        self._init_weights()
        self.to(self.device)

    def _init_weights(self) -> None:
        """Initialize weights of the model."""
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)
        if self.has_outproj:
            nn.init.xavier_uniform_(self.w_o.weight)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool = True,
        inference: bool = False,
    ) -> torch.Tensor:
        """
        Calculate the attention mechanism output.
        All below tensor are described using this nomenclature: B - batch,
        L - sequence length/attention dimension, H -  number of heads, D - head dim,
        M - number of random features, E - embedding dimension

        :param q: query tensor of size [B, L, E]
        :param key: key tensor of size [B, L, E]
        :param value: value tensor of size [B, L, E]
        :param causal: flag determining whether to use causal attention
        :param inference: flag determining whether to use inference mode

        :return: attention mechanism output
        """
        # Linear projections
        query = self.w_q(query)
        key = self.w_k(key)
        value = self.w_v(value)

        # Apply activation function
        if self.act_fun is not None:
            query = self.act_fun(query)
            key = self.act_fun(key)

        # Apply Multi-Head Attention mechanism
        attention_result = self.attention_mechanism(
            query, key, value, causal=causal, inference=inference
        )

        # Output projection with dropout
        if self.has_outproj:
            attention_result = self.dropout(self.w_o(attention_result))

        return attention_result
