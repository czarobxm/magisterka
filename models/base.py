"""Base model class."""

from typing import Union

import torch
from torch import nn

from transformer.layers.multi_head_attention.attention_mechanism.attn_params import (
    LinearAttnParams,
    VanillaParams,
    PerformerParams,
    CosformerParams,
)


class BaseModel(nn.Module):
    """Base model class.

    This class is used as a base class for all models in the project.
    :param d_model: dimension of embedding, size of one vector in a sequence
    :param vocab_size: size of the vocabulary
    :param structure: structure of the model
    :param num_heads: number of heads
    :param method_params: dictionary with method parameters
    :param apply_rotary_pos_enc: flag determining whether to use rotary position encoding
    :param dropout: dropout rate
    :param max_length: maximum length of the sequence
    :param attn_has_outproj: flag determining whether to use output projection
    :param act_fun: activation function
    :param norm_before: flag determining whether to use PostNorm or PreNorm
    :param pos_enc_type: type of positional encoding
    :param use_embedding: flag determining whether to use embedding
    :param device: device
    """

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
        ],
        apply_rotary_pos_enc: bool,
        dropout: float,
        attn_has_outproj: bool,
        act_fun: str,
        post_norm: bool,
        pos_enc_type: str,
        use_embedding: bool,
        device: str,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.structure = structure
        self.num_heads = num_heads
        self.method_params = method_params
        self.apply_rotary_pos_enc = apply_rotary_pos_enc
        self.dropout = dropout
        self.attn_has_outproj = attn_has_outproj
        self.pos_enc_type = pos_enc_type
        self.act_fun_name = act_fun
        if act_fun == "gelu":
            self.act_fun = nn.GELU()
        elif act_fun == "relu":
            self.act_fun = nn.ReLU()
        elif act_fun == "none":
            self.act_fun = None
        self.post_norm = post_norm
        self.use_embedding = use_embedding
        self.device = device

        self.n_layers, self.sizes = self.parse_structure()

    def forward(self, x: torch.Tensor):
        """Forward pass."""
        raise NotImplementedError

    def parse_structure(self):
        """Parse the structure string and return the number of layers and sizes."""
        blocks = self.structure.split(",")
        n_layers = []
        sizes = []
        for block in blocks:
            n_l, size = block.split("x")
            n_layers.append(int(n_l))
            sizes.append(int(size))
        return n_layers, sizes

    def get_hyperparams(self):
        """Return hyperparameters of the model."""
        params_dict = {
            # Model parameters
            "structure": self.structure,
            "vocab_size": self.vocab_size,
            "use_embedding": self.use_embedding,
            "embedder_type": "learnable",
            "pos_enc_type": self.pos_enc_type,
            "number_of_params": sum(p.numel() for p in self.parameters()),
            "device": self.device,
            # MHA parameters
            "mha_type": self.method_params.method,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "attn_has_outproj": self.attn_has_outproj,
            "act_fun": self.act_fun_name,
            "apply_rotary_pos_enc": self.apply_rotary_pos_enc,
            "post_norm": self.post_norm,
        }

        return params_dict
