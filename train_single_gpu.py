from typing import Dict, Any
import argparse
import hashlib
import logging
import signal
import time

import neptune
import torch
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer

from training_func import train
from utils import create_dataloaders, initialize_model
from attention_params import get_attention_params
from utils import setup_logging

from config import (
    SPECIAL_TOKENS_DICT,
    DEFAULT_ARGS,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training parameters")

    # Add arguments based on DEFAULT_ARGS
    for arg, value in DEFAULT_ARGS.items():
        arg_type = type(value) if value is not None else str
        if isinstance(value, bool):
            parser.add_argument(f"--{arg}", action="store_true", help="")
        elif isinstance(value, list):
            parser.add_argument(
                f"--{arg}", nargs="*", type=arg_type, default=value, help=""
            )
        else:
            parser.add_argument(f"--{arg}", type=arg_type, default=value, help="")

    return parser.parse_args()


def setup_neptune(args: argparse.Namespace) -> neptune.Run:
    run = neptune.init_run(
        project=args.project,
        api_token=args.api_token,
        custom_run_id=hashlib.md5(str(time.time()).encode()).hexdigest(),
        name=args.name,
        tags=args.tags,
    )

    def handler(sig, frame):  # pylint: disable=unused-argument
        run.stop()

    signal.signal(signal.SIGINT, handler)
    return run


def setup_tokenizer(args: argparse.Namespace) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, resume_download=None)
    if args.tokenizer in ["gpt2", "bert-base-uncased"]:
        tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
        tokenizer._tokenizer.post_processor = (  # pylint: disable=protected-access
            TemplateProcessing(
                single=f"{tokenizer.bos_token} $A {tokenizer.eos_token}",
                special_tokens=[
                    (tokenizer.eos_token, tokenizer.eos_token_id),
                    (tokenizer.bos_token, tokenizer.bos_token_id),
                ],
            )
        )
    return tokenizer


def setup_training(args: argparse.Namespace, model: torch.nn.Module) -> Dict[str, Any]:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = (
        torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            gamma=args.scheduler_gamma,
            milestones=args.scheduler_milestones,
        )
        if args.scheduler
        else None
    )
    loss_fn = (
        torch.nn.CrossEntropyLoss()
        if args.criterion == "cross_entropy"
        else torch.nn.BCEWithLogitsLoss()
    )
    return {"optimizer": optimizer, "scheduler": scheduler, "loss_fn": loss_fn}


def main():
    args = parse_arguments()
    setup_logging()

    logging.info("Starting training script.")

    tokenizer = setup_tokenizer(args)
    logging.info("Tokenizer set up.")

    logging.info("Loading and tokenizing dataset...")
    train_loader, val_loader, test_loader, num_classes = create_dataloaders(
        args, tokenizer
    )
    logging.info("Data loaders created.")

    run = setup_neptune(args)
    logging.info("Neptune run initialized.")

    method_params = get_attention_params(args)
    model = initialize_model(args, tokenizer, num_classes, method_params)
    logging.info("Model %s initialized.", args.model)

    training_setup = setup_training(args, model)
    logging.info("Training setup completed.")

    logging.info("Starting training...")
    train(
        model,
        args,
        training_setup["optimizer"],
        training_setup["scheduler"],
        training_setup["loss_fn"],
        train_loader,
        val_loader,
        test_loader,
        run,
        task=args.task,
        epochs=args.epochs,
    )
    logging.info("Training finished.")

    logging.info("Average loss on test set: %s", run["metrics/test_avg_loss"])
    logging.info("Accuracy on test set: %s", "metrics/test_acc")
    logging.info("Evaluation finished.")

    run.stop()


main()
