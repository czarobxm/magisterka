from torch import nn


class BasePositionalEncoding(nn.Module):
    def __init__(self, max_length, d_model):
        super().__init__()
        self.max_length = max_length
        self.d_model = d_model

    def forward(self, x):
        raise NotImplementedError
