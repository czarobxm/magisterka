from typing import Union, List, Optional

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
        post_norm: bool = False,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        super().__init__()
        self.d_model = d_model
        self.sizes = sizes
        self.device = device

        self._validate_inputs(n_layers, sizes)

        self.downsampling_layers, self.upsampling_layers = self._create_sampling_layers()
        self.shift_right_layers = self._create_shift_right_layers()
        self.decoder_chunks = self._create_decoder_chunks(
            n_layers,
            num_heads,
            method_params,
            apply_rotary_pos_enc,
            dropout,
            act_fun,
            post_norm,
            has_outproj,
        )

        self.to(device)

    def _validate_inputs(self, n_layers: List[int], sizes: List[int]) -> None:
        """Validate input parameters."""
        assert len(n_layers) == len(sizes), "n_layers and sizes must have the same length"
        for i in range(len(sizes) - 1):
            if sizes[i] > sizes[i + 1]:
                assert sizes[i] % sizes[i + 1] == 0, "Adjacent sizes must be divisible"
            else:
                assert sizes[i + 1] % sizes[i] == 0, "Adjacent sizes must be divisible"

    def _create_sampling_layers(self) -> nn.ModuleList:
        """Create downsampling and upsampling layers."""
        downsampling_layers = nn.ModuleList()
        upsampling_layers = nn.ModuleList()
        for i in range(len(self.sizes) - 1):
            if self.sizes[i] > self.sizes[i + 1]:
                factor = self.sizes[i] // self.sizes[i + 1]
                downsampling_layers.append(DownsamplingLayer(self.d_model, factor))
            else:
                factor = self.sizes[i + 1] // self.sizes[i]
                upsampling_layers.append(UpsamplingLayer(self.d_model, factor))
        return downsampling_layers, upsampling_layers

    def _create_shift_right_layers(self) -> nn.ModuleList:
        """Create shift right layers."""
        shift_right_layers = nn.ModuleList()
        for i in range(len(self.sizes) - 1):
            if self.sizes[i] > self.sizes[i + 1]:
                factor = self.sizes[i] // self.sizes[i + 1]
                shift_right_layers.append(ShiftRight(shift=factor - 1))
        return shift_right_layers

    def _create_decoder_chunks(
        self,
        n_layers: List[int],
        num_heads: int,
        method_params: Union[
            LinearAttnParams, VanillaParams, PerformerParams, CosformerParams
        ],
        apply_rotary_pos_enc: bool,
        dropout: float,
        act_fun: Optional[nn.Module],
        post_norm: bool,
        has_outproj: bool,
    ) -> nn.ModuleList:
        """Create decoder chunks."""
        return nn.ModuleList(
            [
                Block(
                    n_layers=n,
                    d_model=self.d_model,
                    num_heads=num_heads,
                    method_params=method_params,
                    apply_rotary_pos_enc=apply_rotary_pos_enc,
                    dropout=dropout,
                    has_outproj=has_outproj,
                    act_fun=act_fun,
                    post_norm=post_norm,
                    device=self.device,
                )
                for n in n_layers
            ]
        )

    def forward(
        self, x: torch.Tensor, causal: bool = True, inference: bool = False
    ) -> torch.Tensor:
        residuals = []

        n_downsampling_layers = len(self.downsampling_layers)

        # Downsampling path
        for i, (dec, downsample) in enumerate(
            zip(
                self.decoder_chunks[:n_downsampling_layers],
                self.downsampling_layers,
            )
        ):
            x = dec(x, causal=causal, inference=inference)
            residuals.append(x)
            x = self.shift_right_layers[i](x)
            x = downsample(x)

        # Middle block
        x = self.decoder_chunks[n_downsampling_layers](
            x, causal=causal, inference=inference
        )

        # Upsampling path
        for dec, upsample, residual in zip(
            self.decoder_chunks[n_downsampling_layers:],
            self.upsampling_layers,
            reversed(residuals),
        ):
            x = upsample(x)
            x = x + residual
            x = dec(x, causal=causal, inference=inference)

        return x
