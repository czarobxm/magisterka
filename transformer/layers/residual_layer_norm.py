from torch import nn


class ResidualLayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, _x, x, alpha=1):
        return self.norm(alpha * _x + x)
