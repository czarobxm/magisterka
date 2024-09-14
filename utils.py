from argparse import Namespace
from typing import Union

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data import TextClassificationDataset, TextGenerationDataset
from models import ClassifierTransformer, DecoderOnlyTransformer
from transformer.layers.multi_head_attention.attention_mechanism.attn_params import (
    CosformerParams,
    PerformerParams,
    VanillaParams,
)


def create_dataloaders(args: Namespace, tokenizer: AutoTokenizer):
    num_classes = None

    if args.task == "classification":
        train_ds = TextClassificationDataset(
            args.dataset,
            split="train",
            tokenizer=tokenizer,
            max_length=args.max_length,
            prepare_dataset=True,
        )
        val_ds = TextClassificationDataset(
            args.dataset,
            split="val",
            tokenizer=tokenizer,
            max_length=args.max_length,
            prepare_dataset=True,
        )
        test_ds = TextClassificationDataset(
            args.dataset,
            split="test",
            tokenizer=tokenizer,
            max_length=args.max_length,
            prepare_dataset=True,
        )

        train_loader = DataLoader(train_ds, batch_size=args.batch_size)
        if val_ds.texts is None:
            val_loader = DataLoader(test_ds, batch_size=args.batch_size)
        else:
            val_loader = DataLoader(val_ds, batch_size=args.batch_size)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size)
        del train_ds, val_ds, test_ds

        num_classes = 2

    elif args.task == "sequence_modelling" or args.task == "image_generation":
        train_ds = TextGenerationDataset(
            args.dataset,
            split="train",
            tokenizer=tokenizer,
            max_length=args.max_length,
            device=args.device,
        )
        val_ds = TextGenerationDataset(
            args.dataset,
            split="val",
            tokenizer=tokenizer,
            max_length=args.max_length,
            device=args.device,
        )
        test_ds = TextGenerationDataset(
            args.dataset,
            split="test",
            tokenizer=tokenizer,
            max_length=args.max_length,
            device=args.device,
        )

        train_loader = DataLoader(train_ds, batch_size=args.batch_size)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size)

        del train_ds, val_ds, test_ds
    else:
        raise NotImplementedError(f"Task {args.task} not implemented.")

    return train_loader, val_loader, test_loader, num_classes


def initialize_model(
    args: Namespace,
    tokenizer: AutoTokenizer,
    num_classes: int,
    method_params: Union[CosformerParams, PerformerParams, VanillaParams],
):
    if args.model == "classifier":
        model = ClassifierTransformer(
            d_model=args.d_model,
            vocab_size=len(tokenizer),
            structure=args.structure,
            num_classes=num_classes,
            num_heads=args.num_heads,
            method_params=method_params,
            apply_rotary_pos_enc=args.apply_rotary_pos_enc,
            dropout=args.dropout,
            attn_has_outproj=args.has_outproj,
            act_fun=args.act_fun,
            post_norm=args.post_norm,
            device=args.device,
        )
    elif args.model == "decoder_only":
        model = DecoderOnlyTransformer(
            d_model=args.d_model,
            vocab_size=len(tokenizer),
            structure=args.structure,
            num_heads=args.num_heads,
            method_params=method_params,
            apply_rotary_pos_enc=args.apply_rotary_pos_enc,
            dropout=args.dropout,
            attn_has_outproj=args.has_outproj,
            act_fun=args.act_fun,
            post_norm=args.post_norm,
            use_embedding=False if args.task == "image_generation" else True,
            device=args.device,
        )
    else:
        raise ValueError(f"Model {args.model} not implemented.")

    return model
