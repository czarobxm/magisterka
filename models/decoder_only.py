"""Module containing the transformer-decoder-only model."""

from typing import Union

import torch
from torch import nn

from models.base import BaseModel
from transformer.blocks.block import Block
from transformer.blocks.hourglass import HourglassBlock
from transformer.positional_encoding import PositionalEncoding
from transformer.layers.multi_head_attention.attention_mechanism.attn_params import (
    LinearAttnParams,
    VanillaParams,
    PerformerParams,
    CosformerParams,
)
from transformer.blocks.utils import ShiftRight


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
        attn_has_outproj: bool = True,
        act_fun: str = None,
        norm_before: bool = False,
        pos_enc_type: str = "learnable",
        use_embedding: bool = True,
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
            attn_has_outproj=attn_has_outproj,
            act_fun=act_fun,
            norm_before=norm_before,
            pos_enc_type=pos_enc_type,
            use_embedding=use_embedding,
            device=device,
        )

        # Embedders
        self.embedder = nn.Embedding(self.vocab_size, self.d_model)

        # Positional Encodings
        self.pos_enc = PositionalEncoding(
            self.sizes[0], self.d_model, self.pos_enc_type, device=self.device
        )
        # Shift right
        self.shift_right = ShiftRight(shift=1)
        # Encoder
        if len(self.n_layers) == len(self.sizes) == 1:
            self.decoder_block = Block(
                n_layers=self.n_layers,
                d_model=self.d_model,
                num_heads=self.num_heads,
                method_params=self.method_params,
                apply_rotary_pos_enc=self.apply_rotary_pos_enc,
                dropout=self.dropout,
                has_outproj=self.attn_has_outproj,
                act_fun=self.act_fun,
                norm_before=self.norm_before,
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
                has_outproj=self.attn_has_outproj,
                act_fun=self.act_fun,
                norm_before=self.norm_before,
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
        # Encoder
        x = self.decoder_block(x, causal=True, inference=inference)
        # Linear
        x = self.classifier(x)
        return x
