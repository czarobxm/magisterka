# pylint: disable=no-member
"""
Cosformer attention implementation based on an official implementation: 
https://github.com/OpenNLPLab/cosFormer/blob/main/cosformer.py
"""
from typing import List

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
    [B, Nh, L, Dh] -> [B, Nh, L, 2 * Dh]
    """
    x = torch.cat(
        [
            x
            * torch.sin(
                weight_index[:, :seq_len, :] / m  # pylint: disable=unsubscriptable-object
            ),
            x
            * torch.cos(
                weight_index[:, :seq_len, :] / m  # pylint: disable=unsubscriptable-object
            ),
        ],
        dim=-1,
    )
    return x


class Cosformer(BaseAttentionMechanism):
    """
    Cosformer linear attention mechanism - https://arxiv.org/abs/2202.08791.

    This class is designed to provide efficient non-causal attention for deep learning
    models and supports multiple hardware platforms:
    - MPS: Implementation based on the official repository
    (https://github.com/OpenNLPLab/cosFormer).
    - CPU and CUDA: Implementation based on causal_dot_product function
    from FastTransformers (https://github.com/idiap/fast-transformers).

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
        self.kv.zero_()
        self.k_.zero_()

    def inference(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """Constant time inference for the linear attention model."""
        kv_ = torch.einsum(
            "nld,nlm->ndm", key, value
        )  # [B, L, 2 * Dh], [B, L, Dh] -> [B, 2 * Dh, Dh]

        # Update internal states
        self.kv = self.kv + kv_  # pylint: disable=attribute-defined-outside-init
        self.k_ = self.k_ + key[:, 0, :]  # pylint: disable=attribute-defined-outside-init

        z_ = 1 / torch.clamp_min(
            torch.einsum("nld,nd->nl", query, self.k_), self.eps
        )  # [B, L, 2 * Dh], [B, 2 * Dh] -> [B, L]
        attn_output = torch.einsum(
            "nld,ndm,nl->nlm", query, self.kv, z_
        )  # [B, L, 2 * Dh], [B, 2 * Dh, Dh], [B, L, Dh]  -> [B, L, Dh]
        return attn_output

    def multihead_reshape(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Reshape the input tensors for the multi-head attention mechanism. The reshape size
        depends on the hardware platform.
        """
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
        All below tensor are described using this nomenclature: B - batch,
        L - query sequence length/attention dimension,
        S - value and key sequence length/attention dimension, H -  number of heads,
        D - head dim.

        :param q: query tensor of size [B, Nh, L, Hd] for device in ["cuda", "cpu"]
            or [B * Nh, L, Hd] for other cases
        :param key: key tensor of size [B, Nh, L, Hd] for device in ["cuda", "cpu"]
            or [B * Nh, L, Hd] for other cases
        :param value: value tensor of size [B, Nh, L, Hd] for device in ["cuda", "cpu"]
            or [B * Nh, L, Hd] for other cases

        :return: attention mechanism output, tensor of shape [B, L, D]
        """
        # Get sizes of the input tensors
        batch_size = query.size(0)
        tgt_len = query.size(2)
        src_len = key.size(2)

        # Apply feature map to the query and key
        q_, k_ = self.feature_map(query, key, tgt_len, src_len, start_pos)

        # Apply causal or non causal attention mechanism or inference
        if causal and not inference:
            out = attention_causal(
                query=q_,
                key=k_,
                value=value,
                num_heads=self.num_heads,
                eps=self.eps,
                kv=self.kv,
                k_=self.k_,
                device=self.device,
            )
        elif not causal and not inference:
            out = attention_noncausal(q_, k_, value, self.eps)
        else:
            raise NotImplementedError("Inference is not supported for the cosformer.")

        # Undo the multi-head reshape. [B, L, Nh, Dh] -> [B, L, D]
        out = undo_multihead_reshape(out, self.d_model, batch_size, tgt_len)

        return out
