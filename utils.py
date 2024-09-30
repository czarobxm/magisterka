import logging
from argparse import Namespace
from typing import Union, Tuple
import math

from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from data.load_data import dataset_loader


from data import TextGenerationDataset
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
    dataset_cls = TextGenerationDataset

    num_classes = 2 if args.task == "classification" else None
    train_data, val_data, test_data = dataset_loader(
        "enwik9", split="all", cache_dir=f"datastorage/{args.dataset}"
    )

    datasets = {
        split: dataset_cls(
            dataset=data,
            split=split,
            tokenizer=tokenizer,
            max_length=args.max_length,
            device=args.device if args.task != "classification" else None,
        )
        for split, data in zip(
            ["train", "val", "test"], [train_data, val_data, test_data]
        )
    }

    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size)
    val_loader = DataLoader(datasets["val"], batch_size=args.batch_size)
    test_loader = DataLoader(datasets["test"], batch_size=args.batch_size)

    return train_loader, val_loader, test_loader, num_classes


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
