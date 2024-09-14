"""Module containing the classifier transformer-encoder-only model."""

from typing import Union

import torch
from torch import nn

from models.base import BaseModel
from transformer.blocks.block import Block
from transformer.blocks.tightening_block import TighteningBlock
from transformer.positional_encoding import PositionalEncoding
from transformer.layers.multi_head_attention.attention_mechanism.attn_params import (
    LinearAttnParams,
    VanillaParams,
    PerformerParams,
    CosformerParams,
)
from transformer.blocks.utils import ShiftRight


class ClassifierTransformer(BaseModel):
    """Classifier transformer-encoder only model."""

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        structure: str,
        num_classes: int,
        num_heads: int,
        method_params: Union[
            LinearAttnParams,
            VanillaParams,
            PerformerParams,
            CosformerParams,
        ],
        apply_rotary_pos_enc: bool,
        dropout: float,
        attn_has_outproj: bool,
        act_fun: str,
        post_norm: bool,
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
            post_norm=post_norm,
            pos_enc_type=pos_enc_type,
            use_embedding=use_embedding,
            device=device,
        )
        self.num_classes = num_classes

        # Embedders
        self.embedder = nn.Embedding(self.vocab_size, self.d_model)

        # Positional Encodings
        self.pos_enc = PositionalEncoding(
            self.vocab_size, self.d_model, self.pos_enc_type, device=self.device
        )

        # Shift right
        self.shift_right = ShiftRight(shift=1)

        # Encoder
        if len(self.n_layers) == len(self.sizes) == 1:
            self.encoder = Block(
                n_layers=self.n_layers,
                d_model=self.d_model,
                num_heads=self.num_heads,
                method_params=self.method_params,
                apply_rotary_pos_enc=self.apply_rotary_pos_enc,
                dropout=self.dropout,
                has_outproj=self.attn_has_outproj,
                act_fun=self.act_fun,
                post_norm=self.post_norm,
                device=self.device,
            )
        else:
            self.encoder = TighteningBlock(
                d_model=self.d_model,
                n_layers=self.n_layers,
                sizes=self.sizes,
                num_heads=self.num_heads,
                method_params=self.method_params,
                apply_rotary_pos_enc=self.apply_rotary_pos_enc,
                dropout=self.dropout,
                has_outproj=self.attn_has_outproj,
                act_fun=self.act_fun,
                device=self.device,
            )

        # Classifier
        self.classifier = nn.Linear(self.d_model, num_classes)

        # Device
        self.to(device)

    def forward(self, x: torch.Tensor):
        # Shift right
        x = self.shift_right(x)

        # Embedding
        x = self.embedder(x)

        # Positional Encoding
        x = self.pos_enc(x)

        # Encoder
        x = self.encoder(x, causal=False)

        # Pooling
        x = x.mean(dim=-2)

        # Classifier
        x = self.classifier(x)

        return x

    def get_hyperparams(self):
        params_dict = super().get_hyperparams()
        params_dict.update(
            {
                "model": "transformer classifier",
                "num_classes": self.num_classes,
                "class_head": "linear_layer",
            }
        )
        return params_dict
