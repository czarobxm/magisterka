from typing import Dict, Any
import argparse

import torch

from training_setup.scheduler import get_cosine_scheduler_with_warmup


def setup_training(args: argparse.Namespace, model: torch.nn.Module) -> Dict[str, Any]:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    if args.scheduler:
        scheduler = get_cosine_scheduler_with_warmup(
            optimizer,
            num_all_steps=args.scheduler_num_all_steps,
            num_warmup_steps=args.scheduler_lr_warmup_steps,
            final_lr_fraction=args.scheduler_final_lr_fraction,
        )
    else:
        scheduler = None

    loss_fn = (
        torch.nn.CrossEntropyLoss()
        if args.criterion == "cross_entropy"
        else torch.nn.BCEWithLogitsLoss()
    )
    return {"optimizer": optimizer, "scheduler": scheduler, "loss_fn": loss_fn}
