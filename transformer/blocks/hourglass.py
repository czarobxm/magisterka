from typing import Union

import torch
from torch import nn

from transformer.layers.multi_head_attention.attention_mechanism.attn_params import (
    LinearAttnParams,
    VanillaParams,
    PerformerParams,
    CosformerParams,
)
from transformer.blocks.block import Block
from transformer.blocks.utils import ShiftRight, DownsamplingLayer, UpsamplingLayer


class HourglassBlock(nn.Module):
    """
    Hourglass block with downsampling and upsampling layers connected with residual
    connections. Described in https://arxiv.org/pdf/2110.13711.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        sizes: int,
        num_heads: int,
        method_params: Union[
            LinearAttnParams,
            VanillaParams,
            PerformerParams,
            CosformerParams,
        ],
        apply_rotary_pos_enc: bool = True,
        dropout: float = 0.1,
        has_outproj: bool = True,
        act_fun: nn.Module = None,
        norm_before: bool = False,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.sizes = sizes
        self.num_heads = num_heads
        self.dim_head = d_model // num_heads
        self.method_params = method_params
        self.apply_rotary_pos_enc = apply_rotary_pos_enc
        self.dropout = dropout
        self.has_outproj = has_outproj
        self.act_fun = act_fun
        self.norm_before = norm_before
        self.device = device

        downsampling_factors = []
        upsampling_factors = []
        for i in range(len(self.sizes) - 1):
            if self.sizes[i] > self.sizes[i + 1]:
                factor, remainder = divmod(self.sizes[i], self.sizes[i + 1])
                assert remainder == 0, "Sizes must be divisible by each other"
                downsampling_factors.append(factor)
            else:
                factor, remainder = divmod(self.sizes[i], self.sizes[i + 1])
                assert factor == 0, "Sizes must be divisible by each other"
                upsampling_factors.append(self.sizes[i + 1] // self.sizes[i])

        downsampling_layers = nn.ModuleList(
            [DownsamplingLayer(self.d_model, factor) for factor in downsampling_factors]
        )
        upsampling_layers = nn.ModuleList(
            [UpsamplingLayer(self.d_model, factor) for factor in upsampling_factors]
        )

        self.initial_shift_right = ShiftRight(shift=1)
        self.shift_right_layers = nn.ModuleList(
            [ShiftRight(shift=factor - 1) for factor in downsampling_factors]
        )
        self.down_up_sampling_layers = downsampling_layers + upsampling_layers

        assert len(downsampling_layers) == len(
            upsampling_layers
        ), "Number of downsampling and upsampling layers must be equal"

        self.decoder_chunks = nn.ModuleList(
            [
                Block(
                    n_layers=n,
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    method_params=self.method_params,
                    apply_rotary_pos_enc=self.apply_rotary_pos_enc,
                    dropout=self.dropout,
                    has_outproj=False,
                    act_fun=self.act_fun,
                    norm_before=self.norm_before,
                    device=self.device,
                )
                for n, _ in zip(self.n_layers, self.sizes)
            ]
        )
        assert len(self.decoder_chunks) == len(self.down_up_sampling_layers) + 1, (
            "Number of decoder chunks must be equal to the number of the sum of "
            "downsampling and upsampling layers + 1"
        )

    def forward(
        self, x: torch.Tensor, causal: bool = True, inference: bool = False
    ) -> torch.Tensor:
        # Init buffer for residual connections inside the hourglass block
        outputs = []

        # Initial decoder chunk
        x = self.decoder_chunks[0](x, causal=causal, inference=inference)

        for i in range(len(self.decoder_chunks) - 1):
            if isinstance(self.down_up_sampling_layers[i], DownsamplingLayer):
                # 1. Save the output for residual connection
                # 2. Downsample shifted input
                # 3. Pass through decoder chunk
                outputs.append(x)
                x_downsampled = self.down_up_sampling_layers[i](
                    self.shift_right_layers[i](x)
                )
                x = self.decoder_chunks[i + 1](
                    x_downsampled, key_value=x, causal=causal, inference=inference
                )
            else:
                # 1. Upsample the output from the previous decoder chunk
                # 2. Upsample the input and add residual connection
                # 3. Pass through decoder chunk
                x_prime = outputs.pop(-1)
                x_upsampled = x_prime + self.down_up_sampling_layers[i](x)
                x = x_upsampled + self.decoder_chunks[i + 1](
                    x_upsampled, key_value=x_prime, causal=causal, inference=inference
                )

        return x
