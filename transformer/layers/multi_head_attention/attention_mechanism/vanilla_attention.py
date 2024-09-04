from typing import Tuple
import math

import torch
from torch import nn
import torch.nn.functional as F


class VanillaAttention(nn.Module):
    def __init__(self, num_head, embed_dim) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_head
        self.head_dim = embed_dim // num_head

    def multihead_reshape(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Multi-head reashaping - Split heads from size [B, L, D] to size: [B, Nh, L, Dh]
        query = (
            query.contiguous()
            .view(query.size(0), -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key = (
            key.contiguous()
            .view(query.size(0), -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value = (
            value.contiguous()
            .view(query.size(0), -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        return query, key, value

    def scaled_dot_product_attention(self, query, key, value, causal=False):
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1))
        attn_bias = torch.zeros(L, S, dtype=query.dtype)

        if causal:
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        return attn_weight @ value

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool = False,
        inference: bool = False,  # pylint: disable=unused-argument
        eps: float = 1e-12,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """
        Vanilla attention mechanism.
        All below tensor are described using this nomenclature: B - batch, L - sequence length/attention dimension,
        H -  number of heads, D - head dim, M - number of random features
        :param q: query tensor of size [B, H, L, D]
        :param key: key tensor of size [B, H, L, D]
        :param value: value tensor of size [B, H, L, D]

        :return: attention mechanism output
        """
        # Scaled dot product attention
        output = self.scaled_dot_product_attention(query, key, value, causal=causal)

        # Undo multi-head reshaping
        output = output.contiguous().view(query.size(0), -1, self.embed_dim)
        return output
