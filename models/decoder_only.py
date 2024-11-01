"""Module containing the transformer-decoder-only model."""

from typing import Union

import torch
from torch import nn

from models.base import BaseModel
from transformer.blocks.transformer_block import Block
from transformer.blocks.hourglass_block import HourglassBlock
from transformer.positional_encoding import PositionalEncoding
from transformer.multi_head_attention.attention_mechanism.attn_params import (
    LinearAttnParams,
    VanillaParams,
    PerformerParams,
    CosformerParams,
)
from transformer.blocks.hourglass_block.utils import ShiftRight


class DecoderOnlyTransformer(BaseModel):
    """Decoder-only transformer model."""

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        structure: str,
        num_heads: int,
        method_params: Union[
            LinearAttnParams,
            VanillaParams,
            PerformerParams,
            CosformerParams,
        ] = CosformerParams(),
        apply_rotary_pos_enc: bool = True,
        dropout: float = 0.1,
        act_fun: str = None,
        post_norm: bool = False,
        pos_enc_type: str = "learnable",
        use_embedding: bool = True,
        hourglass_downsampling_type: str = "avg",
        hourglass_upsampling_type: str = "linear",
        hourglass_attention_downsampling: bool = True,
        hourglass_attention_upsampling: bool = True,
        hourglass_upsampling_residual: bool = True,
        hourglass_sampling_post_norm: bool = True,
        hourglass_sampling_use_linear: bool = True,
        hourglass_sampling_use_feedforward: bool = True,
        device: str = "cpu",
    ):
        super().__init__(
            d_model=d_model,
            vocab_size=vocab_size,
            structure=structure,
            num_heads=num_heads,
            method_params=method_params,
            apply_rotary_pos_enc=apply_rotary_pos_enc,
            dropout=dropout,
            act_fun=act_fun,
            post_norm=post_norm,
            pos_enc_type=pos_enc_type,
            use_embedding=use_embedding,
            hourglass_downsampling_type=hourglass_downsampling_type,
            hourglass_upsampling_type=hourglass_upsampling_type,
            hourglass_attention_downsampling=hourglass_attention_downsampling,
            hourglass_attention_upsampling=hourglass_attention_upsampling,
            hourglass_upsampling_residual=hourglass_upsampling_residual,
            device=device,
        )

        self.hourglass_sampling_post_norm = hourglass_sampling_post_norm
        self.hourglass_sampling_use_linear = hourglass_sampling_use_linear
        self.hourglass_sampling_use_feedforward = hourglass_sampling_use_feedforward

        # Embedders
        self.embedder = nn.Embedding(self.vocab_size, self.d_model)

        # Positional Encodings
        self.pos_enc = PositionalEncoding(
            self.sizes[0], self.d_model, self.pos_enc_type, device=self.device
        )
        # Shift right
        self.shift_right = ShiftRight(shift=1)
        # Decoder
        if len(self.n_layers) == len(self.sizes) == 1:
            self.decoder_block = Block(
                n_layers=self.n_layers,
                d_model=self.d_model,
                num_heads=self.num_heads,
                method_params=self.method_params,
                apply_rotary_pos_enc=self.apply_rotary_pos_enc,
                dropout=self.dropout,
                act_fun=self.act_fun,
                post_norm=self.post_norm,
                device=self.device,
            )
        else:
            self.decoder_block = HourglassBlock(
                d_model=self.d_model,
                n_layers=self.n_layers,
                sizes=self.sizes,
                num_heads=self.num_heads,
                method_params=self.method_params,
                apply_rotary_pos_enc=self.apply_rotary_pos_enc,
                dropout=self.dropout,
                act_fun=self.act_fun,
                post_norm=self.post_norm,
                downsampling_type=self.hourglass_downsampling_type,
                upsampling_type=self.hourglass_upsampling_type,
                attention_downsampling=self.hourglass_attention_downsampling,
                attention_upsampling=self.hourglass_attention_upsampling,
                upsampling_residual=self.hourglass_upsampling_residual,
                sampling_post_norm=self.hourglass_sampling_post_norm,
                sampling_use_linear=self.hourglass_sampling_use_linear,
                sampling_use_feedforward=self.hourglass_sampling_use_feedforward,
                device=self.device,
            )
        # Classifier
        self.classifier = nn.Linear(self.d_model, self.vocab_size)

        # Device
        self.to(device)

    def forward(self, x: torch.Tensor, inference: bool = False):
        """Produces the output of the decoder block."""
        # Shift right
        x = self.shift_right(x)
        # Embedding
        if self.use_embedding:
            x = self.embedder(x)
        # Positional Encoding
        x = self.pos_enc(x)
        # Decoder
        x = self.decoder_block(x, causal=True, inference=inference)
        # Linear
        x = self.classifier(x)
        return x
