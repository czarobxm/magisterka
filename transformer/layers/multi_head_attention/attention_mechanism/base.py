from torch import nn


class BaseAttentionMechanism(nn.Module):
    def __init__(self, d_model, num_heads) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_head = d_model // num_heads

    def forward(self, query, key, value, causal=False, inference=False):
        raise NotImplementedError("forward method must be implemented")

    def multihead_reshape(self, query, key, value):
        raise NotImplementedError("multihead_reshape method must be implemented")

    def inference(self, query, key, value):
        raise NotImplementedError("inference method must be implemented")
