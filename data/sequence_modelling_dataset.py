"""This module contains dataset class for text generation task."""

import logging


import torch
from transformers import PreTrainedTokenizer


class TextGenerationDataset(torch.utils.data.Dataset):
    """
    A dataset class for text generation task. It loads the dataset and prepares it
    for model training.
    """

    def __init__(
        self,
        dataset: str = None,
        split: str = "train",
        tokenizer: PreTrainedTokenizer = None,
        max_length: int = 512,
        device: str = "cpu",
    ):
        super().__init__()
        self.data = dataset
        logging.info("Dataset loaded")
        self.split = split
        self.device = device

        self.tokenizer = tokenizer
        self.attention_masks = None
        self.token_type_ids = None
        self.max_length = max_length

        self.shuffled_order = torch.randperm(
            len(self.data) // self.max_length + 1
        ).tolist()

    def to(self, device: str):
        """Move data to device"""
        self.data = self.data.to(device)

    def __len__(self):
        return int(len(self.data) // self.max_length) + 1

    def __getitem__(self, index):
        if index > len(self):
            raise IndexError("Index out of range")
        random_index = self.shuffled_order[index]
        start_idx = random_index * self.max_length
        end_idx = (random_index + 1) * self.max_length - 1
        token_dict = self.tokenizer(
            self.data[start_idx:end_idx],
            padding="max_length",
            truncation="longest_first",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return token_dict["input_ids"].squeeze(0)

    def _tolist(self):
        if isinstance(self.data, str):
            print("str")
            return self.data
        self.data = [row["text"] for row in self.data]
        return self.data

    def shuffle(self):
        """Shuffle the dataset"""
        self.data = self.data[torch.randperm(len(self.data))]
        return self.data
