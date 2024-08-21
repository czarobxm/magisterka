import torch


def attention_noncausal(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    Non causal attention mechanism for the cosformer model.

    :param query: query tensor of size [B, Nh, L, Dh]

    :return: attention mechanism output, tensor of shape [B, L, Nh, Dh]
    """
    kv_ = torch.einsum(
        "bnld,bnlm->bndm", key, value
    )  # [B, Nh, L, 2 * Dh], [B, Nh, L, Dh] -> [B, Nh, 2 * Dh, Dh]
    z_ = 1 / torch.clamp_min(
        torch.einsum("bnld,bnd->bnl", query, torch.sum(key, axis=2)), eps
    )  # [B, Nh, L, 2 * Dh], [B, Nh, L] -> [B, Nh, L]
    attn_output = torch.einsum(
        "bnld,bndm,bnl->bnlm", query, kv_, z_
    )  # [B, Nh, L, 2 * Dh], [B, Nh, 2 * Dh, Dh], [B, Nh, L] -> [B, Nh, L, Dh]
    attn_output = attn_output.transpose(1, 2)  # [B, Nh, L, Dh] -> [B, L, Nh, Dh]
    return attn_output
