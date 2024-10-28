"""
Favor+ mechanism described in https://arxiv.org/pdf/2009.14794.pdf.
This implementation is heavily based on the: 
https://github.com/lucidrains/performer-pytorch
and https://github.com/nawnoes/pytorch-performer
"""

from typing import Tuple

import torch
from torch import nn

from fast_transformers.causal_product import causal_dot_product
from transformer.multi_head_attention.attention_mechanism.base import (
    BaseAttentionMechanism,
)
from transformer.multi_head_attention.attention_mechanism.performer.utils import (
    causal_denominator,
    causal_numerator,
    noncausal_denominator,
    noncausal_numerator,
)


def attention_causal_cuda(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, eps: float
) -> torch.Tensor:
    """
    CUDA causal attention mechanism implementation.

    :param query: query tensor of size [B, L, Nh, M]
    :param key: key tensor of size [B, L, Nh, M]
    :param value: value tensor of size [B, L, Nh, Dh]
    """
    # Transpose
    query = torch.transpose(query, 1, 2)  # [B, L, Nh, M] -> [B, Nh, L, M]
    key = torch.transpose(key, 1, 2)  # [B, L, Nh, M] -> [B, Nh, L, M]
    value = torch.transpose(value, 1, 2)  # [B, L, Nh, M] -> [B, Nh, L, M]

    # Compute the normalizers
    denom = 1 / (
        torch.einsum("nlhi,nlhi->nlh", query, key.cumsum(1)) + eps
    )  # [B, Nh, L, M], [B, Nh, L, M] -> [B, Nh, L]

    # Compute the unnormalized result
    kv_context = causal_dot_product(
        query, key, value
    )  # [B, Nh, L, M], [B, Nh, L, M], [B, Nh, L, Dh] -> [B, Nh, L, Dh]
    attn_output = (
        kv_context * denom[:, :, :, None]
    )  # [B, Nh, L, M], [B, Nh, L, 1] -> [B, Nh, L, M]

    attn_output = attn_output.transpose(1, 2)  # [B, Nh, L, Dh] -> [B, L, Nh, Dh]
    return attn_output


def attention_causal_non_cuda(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, device: str
) -> torch.Tensor:
    """
    PyTorch causal attention mechanism implementation.

    :param query: query tensor of size [B, L, Nh, M]
    :param key: key tensor of size [B, L, Nh, M]
    :param value: value tensor of size [B, L, Nh, Dh]
    """
    # Transpose
    query = torch.transpose(query, 0, 1)  # [L,B,Nh,M]
    key = torch.transpose(key, 0, 1)  # [L,B,Nh,M]
    value = torch.transpose(value, 0, 1)  # [L,B,Nh,D]

    av_attn = causal_numerator(
        query, key, value, device=device
    )  # [L, B, Nh, M], [L, B, Nh, M], [L, B, Nh, Dh] -> [L, B, Nh, Dh]
    attn_normalizer = causal_denominator(
        query, key, device=device
    )  # [L, B, Nh, M], [L, B, Nh, M] -> [L, B, Nh]

    av_attn = torch.transpose(av_attn, 0, 1)  # [L, B, Nh, Dh] -> [B, L, Nh, Dh]
    attn_normalizer = torch.transpose(attn_normalizer, 0, 1)  # [L, B, Nh] -> [B, L, Nh]
    attn_normalizer = attn_normalizer.unsqueeze(-1)  # [B, L, Nh] -> [B, L, Nh, 1]

    attention_result = (
        av_attn / attn_normalizer
    )  # [B, L, Nh, Dh] / [B, L, Nh, 1] -> [B, L, Nh, Dh]
    return attention_result


def attention_causal(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, device: str
) -> torch.Tensor:
    """
    Causal Performer attention mechanism.

    :param query: query tensor of size [B, L, Nh, M]
    :param key: key tensor of size [B, L, Nh, M]
    :param value: value tensor of size [B, L, Nh, Dh]

    :return: Causal Performer attention result of shape [B, L, Nh, Dh]
    """
    if device == "cuda" or device == "cpu":
        return attention_causal_cuda(query, key, value, eps=1e-6)
    return attention_causal_non_cuda(query, key, value, device)


def attention_noncausal(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, device: str
) -> torch.Tensor:
    """
    Noncausal Performer attention mechanism.

    :param query: query tensor of size [B, L, Nh, M]
    :param key: key tensor of size [B, L, Nh, M]
    :param value: value tensor of size [B, L, Nh, Dh]

    :return: Noncausal Performer attention result of shape [B, L, Nh, Dh]
    """
    # Transpose
    query = torch.transpose(query, 0, 1)  # [L,B,Nh,M]
    key = torch.transpose(key, 0, 1)  # [L,B,Nh,M]
    value = torch.transpose(value, 0, 1)  # [L,B,Nh,D]

    av_attn = noncausal_numerator(
        query, key, value
    )  # [L, B, Nh, M], [L, B, Nh, M], [L, B, Nh, Dh] -> [L, B, Nh, Dh]
    attn_normalizer = noncausal_denominator(
        query, key, device=device
    )  # [L, B, Nh, M], [L, B, Nh, M] -> [L, B, Nh]

    av_attn = torch.transpose(av_attn, 0, 1)  # [L, B, Nh, Dh] -> [B, L, Nh, Dh]
    attn_normalizer = torch.transpose(attn_normalizer, 0, 1)  # [L, B, Nh] -> [B, L, Nh]
    attn_normalizer = attn_normalizer.unsqueeze(-1)  # [B, L, Nh] -> [B, L, Nh, 1]

    attention_result = (
        av_attn / attn_normalizer
    )  # [B, L, Nh, Dh] / [B, L, Nh, 1] -> [B, L, Nh, Dh]
    return attention_result


class Performer(BaseAttentionMechanism):
    def __init__(
        self,
        num_head,
        d_model,
        kernel_transformation,
        random_features_num,
        random_features_gen_method,
        eps: float = 1e-6,
        device: str = "cpu",
    ) -> None:
        super().__init__(d_model=d_model, num_heads=num_head)
        self.dim_head = self.d_model // num_head
        self.kernel_transformation = kernel_transformation

        self.num_features = random_features_num
        self.random_features = random_features_gen_method(
            self.num_features, self.dim_head
        )
        self.projection_matrix = nn.Parameter(self.random_features, requires_grad=False)

        self.eps = eps
        self.device = device

        self.to(device)

    def multihead_reshape(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Multi-head reashaping - Split heads from size [B, L, D] to size: [B, Nh, L, Dh]
        query = query.view(query.size(0), -1, self.num_heads, self.dim_head).transpose(
            1, 2
        )
        key = key.view(query.size(0), -1, self.num_heads, self.dim_head).transpose(1, 2)
        value = value.view(query.size(0), -1, self.num_heads, self.dim_head).transpose(
            1, 2
        )
        return query, key, value

    def undo_multihead_reshape(self, attn_output: torch.Tensor) -> torch.Tensor:
        """[B, L, Nh, Dh] -> [B, L, D]"""
        batch_size = attn_output.size(0)
        tgt_len = attn_output.size(1)
        return attn_output.contiguous().view(batch_size, tgt_len, -1)

    def inference(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        raise NotImplementedError("Inference not implemented")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool = False,
        inference: bool = False,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """
        Favor attention mechanism described in https://arxiv.org/pdf/2009.14794.pdf.
        All below tensor are described using this nomenclature: B - batch, L - sequence length/attention dimension,
        H -  number of heads, D - head dim, M - number of random features

        :param query: query tensor of size [B, L, H, D]
        :param key: key tensor of size [B, L, H, D]
        :param value: value tensor of size [B, L, H, D]
        :param kernel_transformation: kernel transformation function
        :param causal: flag determining wheter to use causal attention
        :param projection_matrix: random features projection matrix
        :param device: device

        :return: Favor+ normalized attention
        """
        # Transpose matrices after rotational positional encoding
        query = query.transpose(1, 2)  # [B, Nh, L, Dh] -> [B, L, Nh, Dh]
        key = key.transpose(1, 2)  # [B, Nh, L, Dh] -> [B, L, Nh, Dh]
        value = value.transpose(1, 2)  # [B, Nh, L, Dh] -> [B, L, Nh, Dh]

        # Kernel Transformation
        query = self.kernel_transformation(
            query, True, self.projection_matrix
        )  # [B, L, Nh, Dh] -> [B, L, Nh, M]
        key = self.kernel_transformation(
            key, False, self.projection_matrix
        )  # [B, L, Nh, Dh] -> [B, L, Nh, M]

        if causal:
            out = attention_causal(query, key, value, self.device)
        elif not causal:
            out = attention_noncausal(query, key, value, self.device)
        elif not causal and inference:
            out = inference
        else:
            raise ValueError("Invalid attention mode")

        return out
