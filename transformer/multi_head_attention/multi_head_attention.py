"""
Multi Head Attention with different attention mechanisms. On top of that, there is
a possibility to add Linformer's projection before the attention mechanism.
"""

from typing import Union

import torch
from torch import nn
from rotary_embedding_torch import RotaryEmbedding

from transformer.multi_head_attention.attention_mechanism import (
    VanillaAttention,
)
from transformer.multi_head_attention.attention_mechanism import (
    Performer,
)
from transformer.multi_head_attention.attention_mechanism import (
    Cosformer,
)
from transformer.multi_head_attention.attention_mechanism.attn_params import (
    LinearAttnParams,
    VanillaParams,
    PerformerParams,
    CosformerParams,
)


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
        act_fun: nn.Module = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_head = self.d_model // self.num_heads
        self.method_params = method_params
        self.dropout = dropout
        self.apply_rotary_pos_enc = apply_rotary_pos_enc
        self.act_fun = act_fun
        if self.apply_rotary_pos_enc:
            self.rotary_pos_enc = RotaryEmbedding(
                self.dim_head // 2,
            )
        self.device = device

        # Linear transformations
        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)

        # Set attention mechanism
        if self.method_params.method == "vanilla":
            self.attention_mechanism = VanillaAttention(num_heads, d_model=self.d_model)
        if self.method_params.method == "performer":
            self.attention_mechanism = Performer(
                self.num_heads,
                self.d_model,
                kernel_transformation=self.method_params.kernel_transformation,
                random_features_num=self.method_params.random_features_num,
                random_features_gen_method=self.method_params.random_features_gen,
                device=self.device,
            )
        if self.method_params.method == "cosformer":
            self.attention_mechanism = Cosformer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                eps=self.method_params.eps,
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
        query = self.act_fun(query)
        key = self.act_fun(key)

        # Reshape query, key and value to from [B, L, D] to [B, Nh, L, Dh]
        query, key, value = self.attention_mechanism.multihead_reshape(
            query=query, key=key, value=value
        )

        # Apply rotary position encoding
        if self.apply_rotary_pos_enc:
            query = self.rotary_pos_enc.rotate_queries_or_keys(query)
            key = self.rotary_pos_enc.rotate_queries_or_keys(key)

        # Apply Multi-Head Attention mechanism
        attention_result = self.attention_mechanism(
            query, key, value, causal=causal, inference=inference
        )

        # Undo multi-head reshape from [B, Nh, L, Dh] to [B, L, D]
        attention_result = self.attention_mechanism.undo_multihead_reshape(
            attention_result
        )

        return attention_result
