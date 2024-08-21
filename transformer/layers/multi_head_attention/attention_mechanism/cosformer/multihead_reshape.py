import torch


def multihead_reshape(x: torch.Tensor, num_heads: int, dim_head: int) -> torch.Tensor:
    """[B, L, D] -> [B, Nh, L, Dh]"""
    return x.contiguous().view(x.size(0), -1, num_heads, dim_head).transpose(1, 2)


def undo_multihead_reshape(
    x: torch.Tensor, d_model: int, batch_size: int, tgt_len: int
) -> torch.Tensor:
    """[B, L, Nh, Dh] -> [B, L, D]"""
    return x.contiguous().view(batch_size, tgt_len, d_model)
