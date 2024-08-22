"""This module contains dataset class for text classification task."""

from __future__ import annotations
from typing import Tuple

import torch
from transformers import PreTrainedTokenizer

from data.load_data import dataset_loader


class TextClassificationDataset(torch.utils.data.Dataset):
    """
    A dataset class for text classification task. It loads the dataset and prepares it
    for model training.
    """

    def __init__(
        self,
        dataset_name: str = None,
        split: str = "train",
        tokenizer: PreTrainedTokenizer = None,
        max_length: int = 512,
        cache_dir: str = None,
        prepare_dataset: bool = True,
        device: str = "cpu",
    ) -> None:
        self.dataset_name = dataset_name
        if dataset_name is not None:
            self.texts, self.labels = dataset_loader(
                dataset_name, split, cache_dir=cache_dir
            )
        else:
            self.texts, self.labels = None, None
        self.split = split
        self.tokenizer = tokenizer

        self.max_length = max_length

        self.attention_masks = None
        self.device = device

        if prepare_dataset:
            self.prepare_dataset()

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor]:
        text = self.texts[index]
        label = self.labels[index]
        return text, label

    def prepare_dataset(self) -> None:
        """Prepare dataset to be used in model"""
        if self.texts is not None:
            self.tokenize()
            self.shuffle()
            self.to(self.device)

    def tokenize(self) -> Tuple[torch.Tensor]:
        """Use tokenizer to convert text to tokens"""
        if self.tokenizer is None:
            raise ValueError("No tokenizer provided")
        token_dict = self.tokenizer(
            text=self.texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
            return_tensors="pt",
        )
        self.texts = token_dict["input_ids"]
        self.labels = torch.Tensor(self.labels)
        self.attention_masks = token_dict["attention_mask"]
        return self.texts, self.labels

    def shuffle(self) -> Tuple[torch.Tensor]:
        """Shuffle the dataset"""
        indices = torch.randperm(len(self.texts))
        self.texts = self.texts[indices]
        self.labels = self.labels[indices]
        if self.attention_masks is not None:
            self.attention_masks = self.attention_masks[indices]
        return self.texts, self.labels

    def to(self, device="mps") -> None:
        """Move data to device"""
        self.texts = self.texts.to(device)
        self.labels = self.labels.to(device).to(torch.LongTensor)
        self.attention_masks = self.attention_masks.to(device)
