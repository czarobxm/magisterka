# pylint: disable=attribute-defined-outside-init,no-member
"""
Cosformer attention implementation based on the official implementation:
https://github.com/OpenNLPLab/cosFormer/blob/main/cosformer.py
"""

from typing import List, Optional
import math

import torch
from torch.nn import functional as F
from torch import nn
import numpy as np

from transformer.multi_head_attention.attention_mechanism.base import (
    BaseAttentionMechanism,
)
from transformer.multi_head_attention.attention_mechanism.cosformer.attention_noncausal import (
    attention_noncausal,
)
from transformer.multi_head_attention.attention_mechanism.cosformer.attention_causal import (
    attention_causal,
)
from transformer.multi_head_attention.attention_mechanism.cosformer.multihead_reshape import (
    multihead_reshape,
)


def get_index(seq_len: int, start_pos: int) -> nn.Parameter:
    """Create array of indices for the cosformer attention mechanism."""
    index = np.pi / 2 * torch.arange(start_pos, start_pos + seq_len + 1).reshape(1, -1, 1)
    return nn.Parameter(index, requires_grad=False)


def query_key_feature_map(
    x: torch.Tensor, weight_index: torch.Tensor, seq_len: int, m: int
) -> torch.Tensor:
    """
    Compute the query and key feature map for the cosformer attention mechanism.

    Args:
        x (torch.Tensor): Input tensor of shape [B, Nh, L, Dh]
        weight_index (torch.Tensor): Weight index tensor
        seq_len (int): Sequence length
        m (int): Maximum of source and target sequence lengths

    Returns:
        torch.Tensor: Feature map of shape [B, Nh, L, 2 * Dh]
    """
    sin_part = x * torch.sin(weight_index[:, :seq_len, :] / m)
    cos_part = x * torch.cos(weight_index[:, :seq_len, :] / m)
    return torch.cat([sin_part, cos_part], dim=-1)


class Cosformer(BaseAttentionMechanism):
    """
    Cosformer linear attention mechanism - https://arxiv.org/abs/2202.08791.

    This class provides efficient non-causal attention for deep learning models
    and supports multiple hardware platforms:
    - MPS: Implementation based on the official repository
    - CPU and CUDA: Implementation based on causal_dot_product function from FastTransformers
    """

    def __init__(
        self, d_model: int, num_heads: int, eps: float = 1e-6, device: str = "cpu"
    ) -> None:
        """Creates instance and buffers for the cosformer attention mechanism."""
        super().__init__(d_model=d_model, num_heads=num_heads)
        self.eps = eps
        self.device = device
        self.register_buffer(
            "kv",
            torch.zeros(self.num_heads, 2 * self.dim_head, self.dim_head, device=device),
        )
        self.register_buffer(
            "k_", torch.zeros(self.num_heads, 2 * self.dim_head, device=device)
        )
        self.to(device)

    def reset_cache(self) -> None:
        """Reset the internal states of the attention mechanism."""
        self.kv.zero_()  # pylint: disable=no-member
        self.k_.zero_()  # pylint: disable=no-member

    def inference(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """Constant time inference for the linear attention model."""
        kv_ = torch.einsum(
            "nld,nlm->ndm", key, value
        )  # [B, L, 2 * Dh], [B, L, Dh] -> [B, 2 * Dh, Dh]

        # Update internal states
        self.kv = (  # pylint: disable=attribute-defined-outside-init
            self.kv + kv_  # pylint: disable=no-member
        )
        self.k_ = self.k_ + key[:, 0, :]  # pylint: disable=no-member

        # Calculate denominator: [B, L, 2 * Dh], [B, 2 * Dh] -> [B, L]
        z_ = 1 / torch.clamp_min(torch.einsum("nld,nd->nl", query, self.k_), self.eps)

        # Compute attention output: [B, L, 2 * Dh], [B, 2 * Dh, Dh], [B, L] -> [B, L, Dh]
        return torch.einsum("nld,ndm,nl->nlm", query, self.kv, z_)

    def multihead_reshape(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> List[torch.Tensor]:
        """Reshape the input tensors for the multi-head attention mechanism."""
        return (
            multihead_reshape(query, self.num_heads, self.dim_head),
            multihead_reshape(key, self.num_heads, self.dim_head),
            multihead_reshape(value, self.num_heads, self.dim_head),
        )

    def undo_multihead_reshape(self, attn_output: torch.Tensor) -> torch.Tensor:
        """
        Undo the reshape operation for multi-head attention output.

        Args:
            x (torch.Tensor): Input tensor of shape [B, L, Nh, Dh]

        Returns:
            torch.Tensor: Reshaped tensor of shape [B, L, D]

        Where:
            B: Batch size
            Nh: Number of heads
            L: Sequence length
            Dh: Dimension of each head
            D: Model dimension (D = Nh * Dh)
        """
        batch_size, seq_len, num_heads, dim_head = attn_output.size()
        return attn_output.contiguous().view(batch_size, seq_len, num_heads * dim_head)

    def feature_map(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        tgt_len: int,
        src_len: int,
        start_pos: int,
    ) -> List[torch.Tensor]:
        """Feature map for the cosformer attention mechanism."""
        m = max(src_len, tgt_len)
        weight_index = get_index(m, start_pos).to(query)
        q_ = query_key_feature_map(query, weight_index, tgt_len, m)  # [B * Nh, L, 2 * h]
        k_ = query_key_feature_map(key, weight_index, src_len, m)  # [B * Nh, S, 2 * Dh]
        return q_, k_

    # def forward(
    #     self,
    #     query: torch.Tensor,
    #     key: torch.Tensor,
    #     value: torch.Tensor,
    #     causal: bool = True,
    #     inference: bool = False,
    #     start_pos: int = 1,
    # ) -> torch.Tensor:
    #     """
    #     Cosformer attention mechanism - https://arxiv.org/abs/2202.08791

    #     Args:
    #         query (torch.Tensor): Query tensor of shape [B, Nh, L, Dh]
    #         key (torch.Tensor): Key tensor of shape [B, Nh, L, Dh]
    #         value (torch.Tensor): Value tensor of shape [B, Nh, L, Dh]
    #         causal (bool): Whether to use causal attention
    #         inference (bool): Whether to use inference mode
    #         start_pos (int): Starting position for the feature map

    #     Returns:
    #         torch.Tensor: Attention mechanism output of shape [B, L, D]
    #     """
    #     tgt_len, src_len = query.size(2), key.size(2)
    #     q_, k_ = self.feature_map(query, key, tgt_len, src_len, start_pos)

    #     if inference:
    #         raise NotImplementedError("Inference is not supported for the cosformer.")

    #     if causal:
    #         out = attention_causal(q_, k_, value, self.eps, self.kv, self.k_, self.device)
    #     else:
    #         out = attention_noncausal(q_, k_, value, self.eps)

    #     return out

    def forward(self, query, key, value, causal=True, inference=False, start_pos=1):
        Q = F.relu(query)
        K = F.relu(key)

        # apply mask
        # Q = Q.masked_fill(~(mask.to(bool)[:, None, :, None]), 0)
        # K = K.masked_fill(~(mask.to(bool)[:, None, :, None]), 0)

        seq_len = Q.size(2)
        idx = torch.arange(1, seq_len + 1, device=self.device)

        # transform query and key into expanded form
        sin_tr = torch.sin((math.pi / 2) * (idx / seq_len))
        cos_tr = torch.cos((math.pi / 2) * (idx / seq_len))
        q_sin = torch.mul(sin_tr.unsqueeze(-1), Q)
        q_cos = torch.mul(cos_tr.unsqueeze(-1), Q)

        k_sin = torch.mul(sin_tr, K.transpose(-2, -1))
        k_cos = torch.mul(cos_tr, K.transpose(-2, -1))

        # build out d x d intermediate matrices
        attn_inter_sin = torch.matmul(k_sin, value)
        attn_inter_cos = torch.matmul(k_cos, value)

        attn_weights_sin = torch.matmul(q_sin, attn_inter_sin)
        attn_weights_cos = torch.matmul(q_cos, attn_inter_cos)

        # build out normalization
        norm_sin = k_sin.sum(-1).unsqueeze(-1)
        norm_cos = k_cos.sum(-1).unsqueeze(-1)

        norm = torch.matmul(q_sin, norm_sin) + torch.matmul(q_cos, norm_cos)

        # final product for attn scores
        attn = (attn_weights_sin + attn_weights_cos) / norm

        return attn

    def left_product(
        self,
        query: torch.Tensor,
        key: torch.Tensor = None,
        value: torch.Tensor = None,
        causal: bool = True,
        start_pos: int = 1,
    ):
        """
        Left-product Cosformer attention mechanism (of n^2 complexity) - https://arxiv.org/abs/2202.08791

        Args:
            query (torch.Tensor): Query tensor of shape [B, Nh, L, Dh]
            key (torch.Tensor): Key tensor of shape [B, Nh, L, Dh]
            value (torch.Tensor): Value tensor of shape [B, Nh, L, Dh]
            causal (bool): Whether to use causal attention
            inference (bool): Whether to use inference mode
            start_pos (int): Starting position for the feature map

        Returns:
            torch.Tensor: Attention mechanism output of shape [B, L, D]
        """
        tgt_len, src_len = query.size(2), key.size(2)

        q_, k_ = self.feature_map(query, key, tgt_len, src_len, start_pos)
        attn_weight = q_ @ k_.transpose(-2, -1)

        attn_bias = torch.zeros(tgt_len, src_len, dtype=query.dtype, device=query.device)
        if causal:
            causal_mask = torch.triu(
                torch.ones(tgt_len, src_len, dtype=torch.bool, device=query.device),
                diagonal=1,
            )
            attn_bias.masked_fill_(causal_mask, float("-inf"))
            attn_weight = attn_weight.masked_fill(attn_bias == float("-inf"), 0)

        denom = torch.clamp_min(attn_weight.sum(dim=-1, keepdim=True), self.eps)
        attn_weight = attn_weight / denom

        attn_output = attn_weight @ value
        return attn_output.transpose(1, 2)
