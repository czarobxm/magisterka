"""
A class integrating a variety of attention mechanisms.
"""

from typing import Union

import torch
from torch import nn
from rotary_embedding_torch import RotaryEmbedding

from transformer.layers.multi_head_attention.attention_mechanism.attn_params import (
    LinearAttnParams,
    VanillaParams,
    PerformerParams,
    CosformerParams,
)
from transformer.layers.multi_head_attention.attention_mechanism.vanilla_attention import (
    VanillaAttention,
)
from transformer.layers.multi_head_attention.attention_mechanism.performer import (
    Performer,
)
from transformer.layers.multi_head_attention.attention_mechanism.cosformer import (
    Cosformer,
)


class AttentionMechanism(nn.Module):
    """
    Class containing different attention mechanisms (vanilla, linformer,
    performer, cosformer, kerformer). It is prepared to be used in new encoder/decoder
    architecture that projects query to make encoder block's output smaller.
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
        apply_rotary_pos_enc: bool,
        device="cpu",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_head = self.d_model // self.num_heads
        self.method_params = method_params
        self.apply_rotary_pos_enc = apply_rotary_pos_enc
        if self.apply_rotary_pos_enc:
            self.rotary_pos_enc = RotaryEmbedding(
                self.dim_head // 2,
            )
        self.device = device

        # Set attention mechanism
        if self.method_params.method == "vanilla":
            self.attention_mechanism = VanillaAttention(num_heads, embed_dim=self.d_model)
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

        self.to(self.device)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool = False,
        inference: bool = False,
    ) -> torch.Tensor:
        # Reshape query, key and value to from [B, L, D] to [..., L, D]
        query, key, value = self.attention_mechanism.multihead_reshape(
            query=query, key=key, value=value
        )

        # Apply rotary position encoding
        if self.apply_rotary_pos_enc:
            query = self.rotary_pos_enc.rotate_queries_or_keys(query)
            key = self.rotary_pos_enc.rotate_queries_or_keys(key)

        # Apply attention mechanism
        return self.attention_mechanism(
            query=query, key=key, value=value, causal=causal, inference=inference
        )
