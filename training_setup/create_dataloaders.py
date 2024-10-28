from typing import Tuple
from argparse import Namespace

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data import TextGenerationDataset
from data.load_data import dataset_loader


def create_dataloaders(
    args: Namespace, tokenizer: AutoTokenizer
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    dataset_cls = TextGenerationDataset

    split = "all" if args.use_validation else "all_wo_val"

    if args.use_validation:
        train_data, val_data, test_data = dataset_loader(
            "enwik9", split=split, cache_dir=f"datastorage/{args.dataset}"
        )
        names = ["train", "val", "test"]
        sets = [train_data, val_data, test_data]
    elif not args.use_validation:
        train_data, test_data = dataset_loader(
            "enwik9", split=split, cache_dir=f"datastorage/{args.dataset}"
        )
        names = ["train", "test"]
        sets = [train_data, test_data]
    else:
        raise ValueError(
            f"Invalid value for use_validation: {args.use_validation}"
            f"Choose from ['all', 'all_wo_val'] for model training."
        )

    datasets = {
        split: dataset_cls(
            dataset=data,
            split=split,
            tokenizer=tokenizer,
            max_length=args.max_length,
            device=args.device if args.task != "classification" else None,
        )
        for split, data in zip(names, sets)
    }

    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size)
    test_loader = DataLoader(datasets["test"], batch_size=args.batch_size)
    if args.use_validation:
        val_loader = DataLoader(datasets["val"], batch_size=args.batch_size)
        return train_loader, val_loader, test_loader
    return train_loader, None, test_loader
