from typing import Tuple
import math
import torch

from transformer.multi_head_attention.attention_mechanism.base import (
    BaseAttentionMechanism,
)


class VanillaAttention(BaseAttentionMechanism):
    """
    Vanilla attention mechanism.
    """

    def __init__(self, num_heads: int, d_model: int) -> None:
        super().__init__(d_model=d_model, num_heads=num_heads)
        self.head_dim = d_model // num_heads

        if self.head_dim * num_heads != self.d_model:
            raise ValueError(
                f"embed_dim {d_model} not divisible by num_heads {num_heads}"
            )

    def multihead_reshape(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Multi-head reshaping - Split heads from size [B, L, D] to size: [B, Nh, L, Dh]
        """
        batch_size = query.size(0)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        return query, key, value

    def undo_multihead_reshape(self, attn_output: torch.Tensor) -> torch.Tensor:
        """
        Undo multi-head reshaping: Split heads from size [B, Nh, L, Dh] to size [B, L, D]
        """
        batch_size = attn_output.size(0)
        output = attn_output.contiguous().view(batch_size, -1, self.d_model)
        return output

    def scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool = False,
    ) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1))

        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        if causal:
            causal_mask = torch.triu(
                torch.ones(L, S, dtype=torch.bool, device=query.device), diagonal=1
            )
            attn_bias.masked_fill_(causal_mask, float("-inf"))

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        output = attn_weight @ value
        return output.transpose(1, 2)

    def inference(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Inference is not supported for the vanilla attention mechanism."
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool = False,
        inference: bool = False,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """
        Vanilla attention mechanism.

        Args:
            query (torch.Tensor): Query tensor of size [B, Nh, L, Dh]
            key (torch.Tensor): Key tensor of size [B, Nh, L, Dh]
            value (torch.Tensor): Value tensor of size [B, Nh, L, Dh]
            causal (bool): Whether to apply causal mask

        Returns:
            torch.Tensor: Attention mechanism output of size [B, L, D]

        Where:
            B - batch size
            L - sequence length/attention dimension
            D - embedding dimension
        """
        # Scaled dot product attention
        output = self.scaled_dot_product_attention(query, key, value, causal=causal)

        return output
