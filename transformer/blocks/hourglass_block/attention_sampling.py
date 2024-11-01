import torch
from torch import nn

from transformer.feed_forward import FeedForward


class AttentionSampling(nn.Module):
    def __init__(
        self,
        d_model: int,
        factor: int,
        sampling_type: str,
        act_fun: nn.Module,
        use_linear: bool,
        post_norm: bool,
    ) -> None:
        super().__init__()
        self.factor = factor
        self.sampling_type = sampling_type
        self.d_model = d_model
        self.use_linear = use_linear
        self.post_norm = post_norm

        self.act_fun = act_fun
        if self.sampling_type == "downsampling":
            self.attention = self.attention_downsampling
        elif self.sampling_type == "upsampling":
            self.attention = self.attention_upsampling

        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)

        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.ffn = FeedForward(d_model=d_model, hidden=1 * d_model, drop_prob=0.0)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights of the model."""
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)

    def attention_downsampling(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ):
        batch_size, seq_len, d_model = key.size()
        key = key.view(
            batch_size,
            seq_len // self.factor,
            self.factor,
            d_model,
        )
        weights = torch.einsum("bsd,bsfd->bsf", query, key).flatten(1)
        attn_output = torch.einsum("bs,bsd->bsd", weights, value)
        attn_output = attn_output.view(
            batch_size,
            seq_len // self.factor,
            self.factor,
            d_model,
        )
        attn_output = attn_output.sum(dim=2)
        return attn_output

    def attention_upsampling(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ):
        batch_size, seq_len, d_model = query.size()
        query = query.view(
            batch_size,
            seq_len // self.factor,
            self.factor,
            d_model,
        )
        weights = torch.einsum("bsfd,bsd->bsf", query, key)
        attn_output = torch.einsum("bsf,bsd->bsfd", weights, value)
        return attn_output.view(batch_size, seq_len, d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        # Linear projections
        if self.use_linear:
            query = self.w_q(query)
            key = self.w_k(key)
            value = self.w_v(value)

        # Activation functions
        query = self.act_fun(query)
        key = self.act_fun(key)

        # Attention
        if self.post_norm:
            output = self.norm1(query + self.attention(query, key, value))
        else:
            output = query + self.attention(
                self.norm1(query), self.norm1(key), self.norm1(value)
            )

        # Feedforward
        if self.post_norm:
            output = self.norm2(output + self.ffn(output))
        else:
            output = output + self.ffn(self.norm2(output))

        return output
