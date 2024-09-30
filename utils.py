import logging
from argparse import Namespace
from typing import Union, Tuple
import math

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data import TextClassificationDataset, TextGenerationDataset
from models import ClassifierTransformer, DecoderOnlyTransformer
from transformer.layers.multi_head_attention.attention_mechanism.attn_params import (
    CosformerParams,
    PerformerParams,
    VanillaParams,
)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S"
    )


def create_dataloaders(
    args: Namespace, tokenizer: AutoTokenizer
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    dataset_cls = (
        TextClassificationDataset
        if args.task == "classification"
        else TextGenerationDataset
    )
    num_classes = 2 if args.task == "classification" else None
    datasets = {
        split: dataset_cls(
            args.dataset,
            split=split,
            tokenizer=tokenizer,
            max_length=args.max_length,
            cache_dir=f"datastorage/{args.dataset}",
            device=args.device if args.task != "classification" else None,
        )
        for split in ["train", "val", "test"]
    }

    dataloaders = {
        split: DataLoader(dataset, batch_size=args.batch_size)
        for split, dataset in datasets.items()
    }

    # if datasets["val"].texts is None:
    #     dataloaders["val"] = dataloaders["test"]

    return dataloaders["train"], dataloaders["val"], dataloaders["test"], num_classes


def initialize_model(
    args: Namespace,
    tokenizer: AutoTokenizer,
    num_classes: int,
    method_params: Union[CosformerParams, PerformerParams, VanillaParams],
) -> Union[ClassifierTransformer, DecoderOnlyTransformer]:
    common_params = {
        "d_model": args.d_model,
        "vocab_size": len(tokenizer),
        "structure": args.structure,
        "num_heads": args.num_heads,
        "method_params": method_params,
        "apply_rotary_pos_enc": args.apply_rotary_pos_enc,
        "dropout": args.dropout,
        "attn_has_outproj": args.has_outproj,
        "act_fun": args.act_fun,
        "post_norm": args.post_norm,
        "device": args.device,
    }

    if args.model == "classifier":
        return ClassifierTransformer(num_classes=num_classes, **common_params)
    elif args.model == "decoder_only":
        return DecoderOnlyTransformer(
            use_embedding=args.task != "image_generation", **common_params
        )
    else:
        raise ValueError(f"Model {args.model} not implemented.")


def get_cosine_scheduler_with_warmup(
    optimizer, num_warmup_steps, final_lr_fraction, num_all_steps
):
    """
    Function that returns a scheduler that warms up the learning rate for lr_warmup_steps steps,
    then decays it using cosine schedule to final_lr_fraction * lr and then stays constant.
    """

    def get_fraction(step: int):
        if step < num_warmup_steps:
            return (step + 1) / num_warmup_steps
        # cosine schedule that ends at final_lr_fraction * lr, then constant
        elif step < num_all_steps:
            return final_lr_fraction + 0.5 * (1 - final_lr_fraction) * (
                1
                + math.cos(
                    math.pi
                    * (step - num_warmup_steps)
                    / (num_all_steps - num_warmup_steps)
                )
            )
        else:
            return final_lr_fraction

    return LambdaLR(optimizer, get_fraction)
