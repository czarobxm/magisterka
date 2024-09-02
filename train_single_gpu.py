import argparse
import hashlib
import logging
import os
import signal
import time

import neptune
import torch
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer

from training_func import evaluate, train
from utils import create_dataloaders, initialize_model
from transformer.layers.multi_head_attention.attention_mechanism.attn_params import (
    CosformerParams,
    PerformerParams,
    VanillaParams,
)
from transformer.layers.multi_head_attention.attention_mechanism.performer.kernel_transformations import (
    softmax_kernel_transformation,
)
from transformer.layers.multi_head_attention.attention_mechanism.performer.utils import (
    orthogonal_gaussian_random_feature,
)

SPECIAL_TOKENS_DICT = {
    "eos_token": "[EOS]",
    "bos_token": "[BOS]",
    "unk_token": "[UNK]",
    "sep_token": "[SEP]",
    "pad_token": "[PAD]",
    "cls_token": "[CLS]",
    "mask_token": "[MASK]",
}

try:
    NEPTUNE_PROJECT = os.environ["NEPTUNE_PROJECT"]
    NEPTUNE_API_TOKEN = os.environ["NEPTUNE_API_TOKEN"]
except KeyError:
    from neptune_config import NEPTUNE_PROJECT, NEPTUNE_API_TOKEN


def list_of_strings(arg: str):
    """Convert a comma separated string to a list of strings."""
    return arg.split(",")


parser = argparse.ArgumentParser(description="Training parameters")

# Global parameters
parser.add_argument("--task", default="classification", help="")
parser.add_argument("--seed", type=int, default=42, help="")
parser.add_argument("--device", default="cpu", help="")

# Neptune parameters
parser.add_argument("--project", default=NEPTUNE_PROJECT, help="")
parser.add_argument("--api_token", default=NEPTUNE_API_TOKEN, help="")
parser.add_argument("--custom_run_id", default=None, help="")
parser.add_argument("--name", default=None, help="")
parser.add_argument("--tags", default=[], help=None, type=list_of_strings)

# Training parameters
parser.add_argument("--lr", type=float, default=0.00005, help="")
parser.add_argument("--epochs", type=int, default=6, help="")
parser.add_argument("--batch_size", type=int, default=64, help="")
parser.add_argument("--criterion", default="cross_entropy", help="")

# Dataset parameters
parser.add_argument("--dataset", default="imdb", help="")
parser.add_argument("--tokenizer", default="bert-base-uncased", help="")

# Multihead Attention parameters
parser.add_argument("--mha_type", default="vanilla", help="")
parser.add_argument("--d_model", type=int, default=512, help="")
parser.add_argument("--num_heads", type=int, default=8, help="")
parser.add_argument("--dropout", type=int, default=0.1, help="")
parser.add_argument("--max_length", type=int, default=512, help="")
parser.add_argument("--deepnorm", type=bool, default=True, help="")
parser.add_argument("--has_outproj", type=bool, default=True, help="")
parser.add_argument("--act_fun", default="relu", help="")
parser.add_argument("--apply_rotary_pos_enc", action="store_true", help="")
parser.add_argument("--norm_before", type=bool, default=True, help="")

# Performer parameters
parser.add_argument(
    "--kernel_transformation", default="softmax_kernel_transformation", help=""
)
parser.add_argument("--random_features_num", default=128, help="")
parser.add_argument(
    "--random_features_gen", default="orthogonal_gaussian_random_feature", help=""
)

# Kerformer parameters
parser.add_argument("--squeeze_channels", type=int, default=256, help="")

# Linformer parameters
parser.add_argument("--linear_projection_type", default=None, help="")
parser.add_argument("--linear_projection_dim", type=int, default=None, help="")

# Cosformer parameters
parser.add_argument("--eps", type=float, default=1e-6, help="")
# Model type
parser.add_argument("--model", default="classifier", help="")

# Transformer parameterss
parser.add_argument("--pos_enc_type", default="learnable", help="")
parser.add_argument("--structure", default="512x6")

args = parser.parse_args()

# Prepare tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, resume_download=None)
# Add special tokens if word-level tokenizer
if args.tokenizer in ["gpt2", "bert-base-uncased"]:
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
    tokenizer._tokenizer.post_processor = (  # pylint: disable=protected-access
        TemplateProcessing(
            single=tokenizer.bos_token + " $A " + tokenizer.eos_token,
            special_tokens=[
                (tokenizer.eos_token, tokenizer.eos_token_id),
                (tokenizer.bos_token, tokenizer.bos_token_id),
            ],
        )
    )

# Set params for MHA
if args.mha_type == "vanilla":
    method_params = VanillaParams()
elif args.mha_type == "performer":
    if args.kernel_transformation == "softmax_kernel_transformation":
        kernel_transformation = softmax_kernel_transformation
    else:
        raise ValueError(
            f"Kernel transformation {args.kernel_transformation} not implemented."
        )

    if args.random_features_gen == "orthogonal_gaussian_random_feature":
        random_features_gen = orthogonal_gaussian_random_feature
    else:
        raise ValueError(
            f"Random features generator {args.random_features_gen} not implemented."
        )
    method_params = PerformerParams(
        kernel_transformation=kernel_transformation,
        random_features_num=args.random_features_num,
        random_features_gen=random_features_gen,
    )
elif args.mha_type == "cosformer":
    method_params = CosformerParams(eps=args.eps)
else:
    raise NotImplementedError(f"{args.mha_type} attention is not implemented.")


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S"
    )

    logging.info("Starting training script.")
    logging.info("Loading and tokenizing dataset...")
    train_loader, val_loader, test_loader, num_classes = create_dataloaders(
        args, tokenizer
    )
    logging.info("Dataset loaded and tokenized.")

    logging.info("Initializing Neptune run...")
    run = neptune.init_run(
        project=args.project,
        api_token=args.api_token,
        custom_run_id=hashlib.md5(str(time.time()).encode()).hexdigest(),
        name=args.name,
        tags=args.tags,
        # mode="offline",
    )

    def handler(sig, frame):  # pylint: disable=unused-argument
        run.stop()

    signal.signal(signal.SIGINT, handler)
    logging.info("Neptune run initialized.")

    logging.info("Initializing model...")
    model = initialize_model(args, tokenizer, num_classes, method_params)
    logging.info('Model "%s" initialized.', args.model)

    logging.info("Initializing optimizer and loss function...")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.criterion == "cross_entropy":
        loss_fn = torch.nn.CrossEntropyLoss()  # ignore_index=tokenizer.pad_token_id
    if args.criterion == "bin_cross_entropy":
        loss_fn = torch.nn.BCEWithLogitsLoss()

    logging.info("Optimizer and loss function initialized.")

    logging.info("Starting training...")
    train(
        model,
        args,
        optimizer,
        loss_fn,
        train_loader,
        val_loader,
        run,
        task=args.task,
        epochs=args.epochs,
    )
    logging.info("Training finished.")
    logging.info("Evaluating model on test set...")
    avg_loss, acc = evaluate(model, test_loader, loss_fn, run, args.task)
    logging.info("Average loss on test set: %s", avg_loss)
    logging.info("Accuracy on test set: %s", acc)
    logging.info("Evaluation finished.")
    run.stop()


if __name__ == "__main__":
    main()
