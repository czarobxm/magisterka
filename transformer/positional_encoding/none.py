from transformer.positional_encoding.base import BasePositionalEncoding


class NonePositionalEncoding(BasePositionalEncoding):
    def __init__(self, max_length, d_model, device="mps"):
        super().__init__(max_length, d_model)
        self.to(device)

    def forward(self, x):
        return x
