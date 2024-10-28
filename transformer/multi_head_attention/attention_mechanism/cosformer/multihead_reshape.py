import torch


def multihead_reshape(x: torch.Tensor, num_heads: int, dim_head: int) -> torch.Tensor:
    """
    Reshape input tensor for multi-head attention.

    Args:
        x (torch.Tensor): Input tensor of shape [B, L, D]
        num_heads (int): Number of attention heads
        dim_head (int): Dimension of each attention head

    Returns:
        torch.Tensor: Reshaped tensor of shape [B, Nh, L, Dh]

    Where:
        B: Batch size
        L: Sequence length
        D: Model dimension (D = Nh * Dh)
        Nh: Number of heads
        Dh: Dimension of each head
    """
    batch_size, seq_len, _ = x.size()
    return x.contiguous().view(batch_size, seq_len, num_heads, dim_head).transpose(1, 2)
