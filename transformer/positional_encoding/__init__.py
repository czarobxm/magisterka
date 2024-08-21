from torch import nn

from .sinusoidal import SinusoidalPositionalEncoding
from .learnable import LearnablePositionalEncoding
from .none import NonePositionalEncoding


class PositionalEncoding(nn.Module):
    def __init__(self, max_length, d_model, pos_enc_type: str = "rotary", device="cpu"):
        super().__init__()
        self.max_length = max_length
        self.d_model = d_model
        self.device = device
        if pos_enc_type not in ["rotary", "sinusoidal", "learnable", "none"]:
            raise ValueError(f"Invalid embedding type: {pos_enc_type}")
        self.pos_enc_type = pos_enc_type

        if self.pos_enc_type == "sinusoidal":
            self.positional_encoding = SinusoidalPositionalEncoding(
                self.max_length, self.d_model, device=device
            )
        elif self.pos_enc_type == "learnable":
            self.positional_encoding = LearnablePositionalEncoding(
                self.max_length, self.d_model, device=device
            )
        elif self.pos_enc_type == "none":
            self.positional_encoding = NonePositionalEncoding(
                self.max_length, self.d_model, device=device
            )

        self.to(device)

    def forward(self, x):
        return self.positional_encoding(x)
