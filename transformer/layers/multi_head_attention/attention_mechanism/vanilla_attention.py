from typing import Tuple

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
        query = query.view(query.size(0), -1, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        key = key.view(query.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(query.size(0), -1, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        return query, key, value

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
        output = F.scaled_dot_product_attention(  # pylint: disable=not-callable
            query, key, value, is_causal=causal
        )
        # Undo multi-head reshaping
        output = output.contiguous().view(query.size(0), -1, self.embed_dim)
        return output
