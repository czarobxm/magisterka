from typing import Union

from argparse import Namespace

from transformers import AutoTokenizer
from models import DecoderOnlyTransformer
from transformer.multi_head_attention.attention_mechanism.attn_params import (
    CosformerParams,
    PerformerParams,
    VanillaParams,
)


def initialize_model(
    args: Namespace,
    tokenizer: AutoTokenizer,
    method_params: Union[CosformerParams, PerformerParams, VanillaParams],
) -> DecoderOnlyTransformer:
    common_params = {
        "d_model": args.d_model,
        "vocab_size": len(tokenizer),
        "structure": args.structure,
        "num_heads": args.num_heads,
        "method_params": method_params,
        "apply_rotary_pos_enc": args.apply_rotary_pos_enc,
        "dropout": args.dropout,
        "act_fun": args.act_fun,
        "post_norm": args.post_norm,
        "hourglass_downsampling_type": args.hourglass_downsampling_type,
        "hourglass_upsampling_type": args.hourglass_upsampling_type,
        "hourglass_attention_downsampling": args.hourglass_attention_downsampling,
        "hourglass_attention_upsampling": args.hourglass_attention_upsampling,
        "hourglass_upsampling_residual": args.hourglass_upsampling_residual,
        "device": args.device,
    }

    if args.model == "decoder_only":
        return DecoderOnlyTransformer(
            use_embedding=args.task != "image_generation", **common_params
        )
    else:
        raise ValueError(f"Model {args.model} not implemented.")
