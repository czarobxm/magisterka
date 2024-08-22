import torch
from fast_transformers.causal_product import causal_dot_product


def attention_causal_non_cuda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_heads: int,
    eps: float,
    kv_buffer: torch.Tensor,
    k_buffer: torch.Tensor,
) -> torch.Tensor:
    """
    Non CUDA causal attention mechanism for the cosformer model.

    :param query: query tensor of size [B, Nh, L, Dh]

    :return: attention mechanism output, tensor of shape [B, L, Nh, Dh]
    """
    # Create context for every time step
    kv_ = torch.einsum(
        "bnld,bnlm->bnldm", key, value
    )  # [B * Nh, L, 2 * Dh], [B * Nh, L, Dh] -> [B, Nh, L, 2 * Dh, Dh]
    kv_cum = torch.cumsum(kv_, dim=2)

    # Compute the unnormalized result
    qkv = torch.einsum(
        "bnld,bnldm->bnlm", query, kv_cum
    )  # [B * Nh, L, 2 * Dh], [B, Nh, L, 2 * Dh, Dh] -> [B, Nh, L, Dh]

    # Compute the normalizers for each timestep
    k_cum = torch.cumsum(key, dim=2)
    denom = torch.clamp_min(
        torch.einsum("bnlm,bnlm->bnl", query, k_cum), eps
    )  # [B * Nh, 2 * Dh], [B * Nh, 2 * Dh] -> [B, Nh, L]

    # Update internal states
    kv_buffer[:, :, :] = kv_cum[-1, -num_heads:, -1, :, :]
    k_buffer[:, :] = k_cum[-1, -num_heads:, -1, :]

    # Normalize the result
    attn_output = qkv / denom.unsqueeze(
        -1
    )  # [B, Nh, L, Dh], [B, Nh, L, 1] -> [B, Nh, L, Dh]
    attn_output = attn_output.transpose(1, 2)  # [B, Nh, L, Dh] -> [B, L, Nh, Dh]
    return attn_output, kv_buffer, k_buffer


def attention_causal_cuda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    CUDA causal attention mechanism for the cosformer model.

    :param query: query tensor of size [B, Nh, L, Dh]

    :return: attention mechanism output, tensor of shape [B, L, Nh, Dh]
    """
    # Compute the normalizers
    denom = 1 / (
        torch.einsum("nhli,nhli->nhl", query, key.cumsum(2)) + eps
    )  # [B, Nh, L, Dh], [B, Nh, L, Dh] -> [B, Nh, L]

    # Compute the unnormalized result
    kv_context = causal_dot_product(
        query, key, value
    )  # [B, Nh, L, 2 * Dh], [B, Nh, L, 2 * Dh], [B, Nh, L, Dh] -> [B, Nh, L, Dh]
    attn_output = (
        kv_context * denom[:, :, :, None]
    )  # [B, Nh, L, Dh], [B, Nh, L, 1] -> [B, Nh, L, Dh]
    return attn_output.transpose(1, 2)  # [B, Nh, L, Dh] -> [B, L, Nh, Dh]


def attention_causal(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_heads: int,
    eps: float,
    kv: torch.Tensor,
    k_: torch.Tensor,
    device: str,  # pylint: disable=unused-argument
) -> torch.Tensor:
    """
    Causal attention mechanism for the cosformer model.

    :param query: query tensor of size [B, Nh, L, Dh] if self.device is "cuda"
    or "cpu", or [B * Nh, L, Dh] otherwise

    :return: attention mechanism output, tensor of shape [B, L, Nh, Dh]
    """
    if device == "cuda" or device == "cpu":
        return attention_causal_cuda(query, key, value, eps)

    out, kv, k_ = (  # pylint: disable=attribute-defined-outside-init
        attention_causal_non_cuda(query, key, value, num_heads, eps, kv, k_)
    )
    return out
