import torch
from typing import Tuple
from fast_transformers.causal_product import causal_dot_product


def attention_causal_non_cuda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    eps: float,
    kv_buffer: torch.Tensor,
    k_buffer: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Non-CUDA causal attention mechanism for the cosformer model.

    Args:
        query (torch.Tensor): Query tensor of shape [B, Nh, L, Dh]
        key (torch.Tensor): Key tensor of shape [B, Nh, L, Dh]
        value (torch.Tensor): Value tensor of shape [B, Nh, L, Dh]
        num_heads (int): Number of attention heads
        eps (float): Small constant for numerical stability
        kv_buffer (torch.Tensor): Buffer for key-value pairs
        k_buffer (torch.Tensor): Buffer for keys

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - Attention output of shape [B, L, Nh, Dh]
            - Updated kv_buffer
            - Updated k_buffer

    Where:
        B: Batch size
        Nh: Number of heads
        L: Sequence length
        Dh: Dimension of each head
    """
    # Create context for every time step: [B * Nh, L, 2 * Dh], [B * Nh, L, Dh] -> [B, Nh, L, 2 * Dh, Dh]
    kv = torch.einsum("bnld,bnlm->bnldm", key, value)
    kv_cum = torch.cumsum(kv, dim=2)

    # Compute the unnormalized result: [B * Nh, L, 2 * Dh], [B, Nh, L, 2 * Dh, Dh] -> [B, Nh, L, Dh]
    qkv = torch.einsum("bnld,bnldm->bnlm", query, kv_cum)

    # Compute the normalizers for each timestep: [B * Nh, 2 * Dh], [B * Nh, 2 * Dh] -> [B, Nh, L]
    k_cum = torch.cumsum(key, dim=2)
    denom = torch.clamp_min(torch.einsum("bnlm,bnlm->bnl", query, k_cum), eps)

    # TODO: Update internal states
    # kv_buffer[:, :, :] = kv_cum[:, :, -1, :, :]
    # k_buffer[:, :] = k_cum[:, :, -1, :]

    # Normalize the result: [B, Nh, L, Dh], [B, Nh, L, 1] -> [B, Nh, L, Dh]
    attn_output = qkv / denom.unsqueeze(-1)

    # Transpose output to match expected shape: [B, Nh, L, Dh] -> [B, L, Nh, Dh]
    attn_output = attn_output.transpose(1, 2)

    return attn_output, kv_buffer, k_buffer


def attention_causal_cuda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    CUDA causal attention mechanism for the cosformer model.

    Args:
        query (torch.Tensor): Query tensor of shape [B, Nh, L, Dh]
        key (torch.Tensor): Key tensor of shape [B, Nh, L, Dh]
        value (torch.Tensor): Value tensor of shape [B, Nh, L, Dh]
        eps (float): Small constant for numerical stability

    Returns:
        torch.Tensor: Attention output of shape [B, Nh, L, Dh]

    Where:
        B: Batch size
        Nh: Number of heads
        L: Sequence length
        Dh: Dimension of each head
    """
    # Compute the normalizers: [B, Nh, L, Dh], [B, Nh, L, Dh] -> [B, Nh, L]
    denom = 1 / (torch.einsum("bnld,bnld->bnl", query, key.cumsum(2)) + eps)

    # Compute the unnormalized result: [B, Nh, L, 2 * Dh], [B, Nh, L, 2 * Dh], [B, Nh, L, Dh] -> [B, Nh, L, Dh]
    kv_context = causal_dot_product(
        query.contiguous(), key.contiguous(), value.contiguous()
    )

    # Normalize the result: [B, Nh, L, Dh], [B, Nh, L, 1] -> [B, Nh, L, Dh]
    attn_output = kv_context * denom.unsqueeze(-1)
    # Transpose output to match expected shape: [B, Nh, L, Dh] -> [B, L, Nh, Dh]
    return attn_output.transpose(1, 2)


def attention_causal(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    eps: float,
    kv_buffer: torch.Tensor,
    k_buffer: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """
    Causal attention mechanism for the cosformer model.

    Args:
        query (torch.Tensor): Query tensor of shape [B, Nh, L, Dh]
        key (torch.Tensor): Key tensor of shape [B, Nh, L, Dh]
        value (torch.Tensor): Value tensor of shape [B, Nh, L, Dh]
        num_heads (int): Number of attention heads
        eps (float): Small constant for numerical stability
        kv_buffer (torch.Tensor): Buffer for key-value pairs
        k_buffer (torch.Tensor): Buffer for keys
        device (str): Device to run the computation on ('cuda' or 'cpu')

    Returns:
        torch.Tensor: Attention output of shape [B, L, Nh, Dh]

    Where:
        B: Batch size
        Nh: Number of heads
        L: Sequence length
        Dh: Dimension of each head
    """
    if device in ["cuda", "cpu"]:
        return attention_causal_cuda(query, key, value, eps)
    else:
        out, kv_buffer, k_buffer = attention_causal_non_cuda(
            query, key, value, eps, kv_buffer, k_buffer
        )
        return out
