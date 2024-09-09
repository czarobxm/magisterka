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
            LinearAttnParams, VanillaParams, PerformerParams, CosformerParams
        ],
        apply_rotary_pos_enc: bool = True,
        dropout: float = 0.1,
        has_outproj: bool = True,
        act_fun: Union[str, None] = None,
        device: str = "cpu",
    ) -> None:
        """
        Initialize the Tightening block.

        Args:
            d_model (int): Dimension of the model
            n_layers (List[int]): Number of layers for each block
            sizes (List[int]): Sizes for each block
            num_heads (int): Number of attention heads
            method_params (Union[LinearAttnParams, VanillaParams, PerformerParams, CosformerParams]):
                Parameters for the attention method
            apply_rotary_pos_enc (bool): Whether to apply rotary positional encoding
            dropout (float): Dropout probability
            max_length (int): Maximum sequence length
            has_outproj (bool): Whether to include output projection
            act_fun (Union[str, None]): Activation function name
            device (str): Device to run the block on
        """
        super().__init__()
        self.d_model = d_model
        self.sizes = sizes
        self.device = device

        self._validate_inputs(n_layers, sizes)
        self.encoder = self._create_encoder(
            n_layers,
            num_heads,
            method_params,
            apply_rotary_pos_enc,
            dropout,
            has_outproj,
            act_fun,
        )

    def _validate_inputs(self, n_layers: List[int], sizes: List[int]) -> None:
        """Validate input parameters."""
        assert len(n_layers) == len(sizes), "n_layers and sizes must have the same length"
        for i in range(len(sizes) - 1):
            assert sizes[i] % sizes[i + 1] == 0, "Adjacent sizes must be divisible"

    def _create_encoder(
        self,
        n_layers: List[int],
        num_heads: int,
        method_params: Union[
            LinearAttnParams, VanillaParams, PerformerParams, CosformerParams
        ],
        apply_rotary_pos_enc: bool,
        dropout: float,
        has_outproj: bool,
        act_fun: Union[str, None],
    ) -> nn.Sequential:
        """Create the encoder with blocks and downsampling layers."""
        encoder = nn.Sequential()
        for i, n in enumerate(n_layers):
            block = nn.Sequential()
            block.append(
                Block(
                    n_layers=n,
                    d_model=self.d_model,
                    num_heads=num_heads,
                    method_params=method_params,
                    apply_rotary_pos_enc=apply_rotary_pos_enc,
                    dropout=dropout,
                    has_outproj=has_outproj,
                    act_fun=act_fun,
                    device=self.device,
                )
            )
            encoder.append(block)

            if i < len(n_layers) - 1:
                downsample_block = nn.Sequential(
                    nn.Dropout(p=dropout),
                    DownsamplingLayer(self.d_model, self.sizes[i] // self.sizes[i + 1]),
                )
                encoder.append(downsample_block)

        return encoder

    def forward(
        self,
        x: torch.Tensor,
        inference: bool = False,
        causal: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass of the Tightening block.

        Args:
            x (torch.Tensor): Input tensor
            inference (bool): Whether in inference mode
            causal (bool): Whether to use causal attention

        Returns:
            torch.Tensor: Output tensor
        """
        for block in self.encoder:
            if isinstance(block[0], Block):
                x = block[0](x, inference=inference, causal=causal)
            else:
                x = block(x)
        return x
