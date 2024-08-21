"""Module containing tightening encoder block."""

from typing import Union, List

import torch
from torch import nn

from transformer.layers.multi_head_attention.attention_mechanism.attn_params import (
    LinearAttnParams,
    VanillaParams,
    PerformerParams,
    CosformerParams,
)
from transformer.blocks.block import Block
from transformer.blocks.utils import DownsamplingLayer


class TighteningBlock(nn.Module):
    """Tightening encoder block with downsampling layers."""

    def __init__(
        self,
        d_model: int,
        n_layers: List[int],
        sizes: List[int],
        num_heads: int,
        method_params: Union[
            LinearAttnParams,
            VanillaParams,
            PerformerParams,
            CosformerParams,
        ],
        apply_rotary_pos_enc: bool = True,
        dropout: float = 0.1,
        max_length: float = 512,
        has_outproj: bool = True,
        act_fun: str = None,
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
        self.max_length = max_length
        self.has_outproj = has_outproj
        self.act_fun = act_fun
        self.device = device

        # Stack encoder blocks with linear projections
        self.encoder = nn.Sequential()
        block = nn.Sequential()
        for i, n in enumerate(self.n_layers):
            enc = Block(
                n_layers=n,
                d_model=self.d_model,
                num_heads=self.num_heads,
                method_params=self.method_params,
                apply_rotary_pos_enc=self.apply_rotary_pos_enc,
                dropout=self.dropout,
                has_outproj=self.has_outproj,
                act_fun=self.act_fun,
                device=self.device,
            )
            block.append(enc)
            if i < len(self.n_layers) - 1:
                self.encoder.append(block)
                block = nn.Sequential()
                block.append(nn.Dropout(p=self.dropout))
                assert (
                    self.sizes[i] % self.sizes[i + 1] == 0
                ), "Sizes must be divisible by each other"
                block.append(
                    DownsamplingLayer(self.d_model, self.sizes[i] // self.sizes[i + 1])
                )

        self.encoder.append(block)

    def forward(
        self,
        x: torch.Tensor,
        inference: bool = False,
        causal: bool = False,
    ) -> torch.Tensor:
        for block in self.encoder:
            if isinstance(block, Block):
                x = block(x, inference=inference, causal=causal)
            else:
                x = block(x)
        return x
