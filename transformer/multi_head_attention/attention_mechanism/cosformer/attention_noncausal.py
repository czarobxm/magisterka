import torch


def attention_noncausal(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    Non-causal attention mechanism for the cosformer model.

    Args:
        query (torch.Tensor): Query tensor of shape [B, Nh, L, Dh]
        key (torch.Tensor): Key tensor of shape [B, Nh, L, Dh]
        value (torch.Tensor): Value tensor of shape [B, Nh, L, Dh]
        eps (float, optional): Small constant for numerical stability. Defaults to 1e-6.

    Returns:
        torch.Tensor: Attention mechanism output, tensor of shape [B, L, Nh, Dh]

    Where:
        B: Batch size
        Nh: Number of heads
        L: Sequence length
        Dh: Dimension of each head
    """
    # Compute key-value product: [B, Nh, L, 2 * Dh], [B, Nh, L, Dh] -> [B, Nh, 2 * Dh, Dh]
    kv_ = torch.einsum("bnld,bnlm->bndm", key, value)

    # Compute denominator for attention weights: [B, Nh, L, 2 * Dh], [B, Nh, L] -> [B, Nh, L]
    z_ = 1 / torch.clamp_min(
        torch.einsum("bnld,bnd->bnl", query, torch.sum(key, axis=2)), eps
    )

    # Compute attention output: [B, Nh, L, 2 * Dh], [B, Nh, 2 * Dh, Dh], [B, Nh, L] -> [B, Nh, L, Dh]
    attn_output = torch.einsum("bnld,bndm,bnl->bnlm", query, kv_, z_)

    # Transpose output to match expected shape: [B, Nh, L, Dh] -> [B, L, Nh, Dh]
    return attn_output.transpose(1, 2)
