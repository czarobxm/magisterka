"""This module contains dataset class for text generation task."""

import torch
from transformers import PreTrainedTokenizer

from data.load_data import dataset_loader


class TextGenerationDataset(torch.utils.data.Dataset):
    """
    A dataset class for text generation task. It loads the dataset and prepares it
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
    ):
        super().__init__()
        dataset = dataset_loader(dataset_name, split, cache_dir=cache_dir)
        self.data = dataset
        self.split = split
        self.device = device

        self.tokenizer = tokenizer
        self.attention_masks = None
        self.token_type_ids = None
        self.max_length = max_length

        self._tolist()

        if prepare_dataset:
            self.prepare_dataset()

    def to(self, device: str):
        """Move data to device"""
        self.data = self.data.to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def prepare_dataset(self) -> None:
        """Prepare dataset to be used in model"""
        if self.data is not None:
            self.tokenize()
            self.slice_sentences()
            self.pad_sentences()
            self.to(self.device)

    def _tolist(self):
        if isinstance(self.data, str):
            return self.data
        self.data = [row["text"] for row in self.data]
        return self.data

    def tokenize(self) -> torch.Tensor:
        """Use tokenizer to convert text to tokens"""
        if self.tokenizer is None:
            raise ValueError("No tokenizer provided")

        token_dict = self.tokenizer(
            self.data,
            padding="do_not_pad",
            truncation="do_not_truncate",
            return_tensors="np",
        )

        data = token_dict["input_ids"].tolist()
        self.data = [torch.Tensor(sentence) for sentence in data]
        return self.data

    def slice_sentences(self) -> torch.Tensor:
        """Slice sentences into blocks of max_length"""
        sliced_sentences = []
        block_length = self.max_length + 1
        last_chunk = None
        for sentence in self.data:
            if last_chunk is not None:
                sentence = torch.cat([last_chunk, sentence])
            n_chunks = len(sentence) // block_length
            last_chunk_len = len(sentence) % block_length
            if last_chunk_len == 0:
                slices_sizes = [block_length] * n_chunks
                last_chunk = None
            else:
                slices_sizes = [block_length] * n_chunks + [last_chunk_len]
            sliced_sentences.extend(torch.split(sentence, slices_sizes))
            last_chunk = sliced_sentences.pop(-1)
        self.data = sliced_sentences

        return self.data

    def pad_sentences(self) -> torch.Tensor:
        """Pad sentences to max_length"""
        self.data = self.tokenizer.pad({"input_ids": self.data}, padding="longest")[
            "input_ids"
        ].to(torch.long)
        return self.data
