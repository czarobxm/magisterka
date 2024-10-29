import torch
from torch import nn

from transformer.feed_forward import FeedForward


class AttentionDownsampling(nn.Module):
    def __init__(self, d_model, downsampling_factor, act_fun: nn.Module) -> None:
        super().__init__()
        self.downsampling_factor = downsampling_factor
        self.d_model = d_model

        self.act_fun = act_fun

        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)

        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.avg_pool = nn.AvgPool1d(
            kernel_size=downsampling_factor, stride=downsampling_factor
        )
        self.ffn = FeedForward(d_model=d_model, hidden=4 * d_model, drop_prob=0.0)

    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        batch_size, seq_len, d_model = key.size()
        key = key.view(
            batch_size,
            seq_len // self.downsampling_factor,
            self.downsampling_factor,
            d_model,
        )
        weights = torch.einsum("bsd,bsfd->bsf", query, key).flatten(1)
        attn_output = torch.einsum("bs,bsd->bsd", weights, value)
        return self.avg_pool(attn_output.transpose(-1, -2)).transpose(-2, -1)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        # Linear projections
        query = self.w_q(query)
        key = self.w_k(key)
        value = self.w_v(value)

        # Activation functions
        query = self.act_fun(query)
        key = self.act_fun(key)

        # Attention
        attn_output = query + self.attention(
            self.norm1(query), self.norm1(key), self.norm1(value)
        )
        # Feedforward
        ffn_output = attn_output + self.ffn(self.norm2(attn_output))

        return ffn_output


class AttentionUpsampling(nn.Module):
    def __init__(self, d_model, upsampling_factor, act_fun: nn.Module) -> None:
        super().__init__()
        self.upsampling_factor = upsampling_factor
        self.d_model = d_model

        self.act_fun = act_fun

        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)

        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)

        self.ffn = FeedForward(d_model=d_model, hidden=4 * d_model, drop_prob=0.0)

    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        batch_size, seq_len, d_model = query.size()
        query = query.view(
            batch_size,
            seq_len // self.upsampling_factor,
            self.upsampling_factor,
            d_model,
        )

        weights = torch.einsum("bsfd,bsd->bsf", query, key)
        attn_output = torch.einsum("bsf,bsd->bsfd", weights, value)
        return attn_output.view(batch_size, seq_len, d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        # Linear projections
        query = self.w_q(query)
        key = self.w_k(key)
        value = self.w_v(value)

        # Activation functions
        query = self.act_fun(query)
        key = self.act_fun(key)

        # Attention
        attn_output = query + self.attention(
            self.norm1(query), self.norm1(key), self.norm1(value)
        )
        # Feedforward
        ffn_output = attn_output + self.ffn(self.norm2(attn_output))
        return ffn_output
