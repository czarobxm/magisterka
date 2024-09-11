# pylint: disable=attribute-defined-outside-init,no-member
"""
Cosformer attention implementation based on the official implementation:
https://github.com/OpenNLPLab/cosFormer/blob/main/cosformer.py
"""

from typing import List, Tuple

import torch
from torch import nn
import numpy as np

from transformer.layers.multi_head_attention.attention_mechanism.base import (
    BaseAttentionMechanism,
)
from transformer.layers.multi_head_attention.attention_mechanism.cosformer.attention_noncausal import (
    attention_noncausal,
)
from transformer.layers.multi_head_attention.attention_mechanism.cosformer.attention_causal import (
    attention_causal,
)
from transformer.layers.multi_head_attention.attention_mechanism.cosformer.multihead_reshape import (
    multihead_reshape,
    undo_multihead_reshape,
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

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool = True,
        inference: bool = False,
        start_pos: int = 1,
    ) -> torch.Tensor:
        """
        Cosformer attention mechanism - https://arxiv.org/abs/2202.08791

        Args:
            query (torch.Tensor): Query tensor of shape [B, Nh, L, Hd]
            key (torch.Tensor): Key tensor of shape [B, Nh, L, Hd]
            value (torch.Tensor): Value tensor of shape [B, Nh, L, Hd]
            causal (bool): Whether to use causal attention
            inference (bool): Whether to use inference mode
            start_pos (int): Starting position for the feature map

        Returns:
            torch.Tensor: Attention mechanism output of shape [B, L, D]
        """
        tgt_len, src_len = query.size(2), key.size(2)
        q_, k_ = self.feature_map(query, key, tgt_len, src_len, start_pos)

        if inference:
            raise NotImplementedError("Inference is not supported for the cosformer.")

        if causal:
            out = attention_causal(q_, k_, value, self.eps, self.kv, self.k_, self.device)
        else:
            out = attention_noncausal(q_, k_, value, self.eps)

        return undo_multihead_reshape(out)
