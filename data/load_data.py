"""
This module contains functions for loading various NLP and image processing datasets.
"""

from typing import Tuple
import os

from datasets import load_dataset


def dataset_loader(dataset_name: str, split: str, cache_dir: str = None) -> Tuple:
    """
    Load dataset based on the dataset name. Train, validation and test splits could be
    returned based on the split parameter. The cache_dir parameter is used to specify
    the directory where the dataset should be stored.
    """
    if dataset_name == "imdb":
        return load_imdb(cache_dir=cache_dir, split=split)
    if dataset_name == "wikitext-103":
        return load_wikitext103(cache_dir=cache_dir, split=split)
    if dataset_name == "enwik8":
        return load_enwik8(cache_dir=cache_dir, split=split)
    if dataset_name == "cifar10":
        return load_cifar10(cache_dir=cache_dir, split=split)
    raise ValueError("Invalid dataset name")


def load_imdb(split: str = "train", cache_dir: str = None):
    """https://huggingface.co/datasets/stanfordnlp/imdb"""
    if cache_dir is None:
        cache_dir = os.path.abspath("./datastorage/imdb")
    if split == "val":
        return None, None
    ds = load_dataset(
        "stanfordnlp/imdb", cache_dir=cache_dir, split=split, resume_download=None
    )
    return ds["text"], ds["label"]


def load_wikitext103(split: str = "train", cache_dir: str = None):
    """https://huggingface.co/datasets/iohadrubin/wikitext-103-raw-v1"""
    ds = load_dataset(
        "iohadrubin/wikitext-103-raw-v1",
        cache_dir=cache_dir,
        split=split,
        resume_download=None,
    )
    return ds


def load_enwik8(split: str = "train", cache_dir: str = None):
    """https://huggingface.co/datasets/LTCB/enwik8"""
    if cache_dir is None:
        cache_dir = os.path.abspath("./datastorage/enwik8/enwik8_data")
    with open(cache_dir, "r", encoding="utf-8") as file:
        ds = file.read()
    if split == "train":
        return ds[:90_000_000]
    if split == "val":
        return ds[90_000_000:95_000_000]
    if split == "test":
        return ds[95_000_000:]
    raise ValueError("Invalid split")


def load_cifar10(split: str = "train", cache_dir: str = None):
    """https://huggingface.co/datasets/uoft-cs/cifar10"""
    if cache_dir is None:
        cache_dir = os.path.abspath("./datastorage/cifar10")
    if split == "val":
        return None, None
    ds = load_dataset(
        "uoft-cs/cifar10", cache_dir=cache_dir, split=split, resume_download=None
    )
    return ds["img"], ds["label"]
