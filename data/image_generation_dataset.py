"""This module contains dataset class for image generation task."""

from typing import Tuple

import torch
import torchvision

from data.load_data import dataset_loader
from matplotlib import pyplot as plt


class ImageGenerationDataset(torch.utils.data.Dataset):
    """
    A dataset class for image generation task. It loads the dataset and prepares it
    for model training.
    """

    def __init__(
        self,
        dataset_name: str = None,
        split: str = "train",
        cache_dir: str = None,
        prepare_dataset: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        dataset = dataset_loader(dataset_name, split, cache_dir=cache_dir)
        self.img = [
            torchvision.transforms.functional.pil_to_tensor(img) for img in dataset[0]
        ]
        self.img = torch.stack(self.img)

        self.label = torch.Tensor(dataset[1])
        self.split = split
        self.device = device

        self.attention_masks = None
        self.token_type_ids = None

        self.normalized = False
        self.flattened = False
        self.original_shape = self.img[0].shape

        if prepare_dataset:
            self.prepare_dataset()

    def normalize(self) -> Tuple[torch.Tensor]:
        """Normalize images - divide by 255."""
        self.img = self.img / 255.0
        self.normalized = True
        return self.img, self.label

    def flatten_images(self) -> Tuple[torch.Tensor]:
        """
        Flatten images.
        [B, C, H, W] -> [B, H, C, W] -> [B, H, C * W]
        """
        self.img = self.img.transpose(1, 2).flatten(2)
        self.flattened = True
        return self.img, self.label

    def to(self, device) -> None:
        """Move data to device"""
        self.img = self.img.to(device)
        self.label = self.label.to(device)

    def show_random_img(self) -> None:
        """Show random image from dataset"""
        img = self.img[torch.randint(0, len(self), (1,)).item()]
        if self.flattened:
            channels = self.original_shape[0]
            height = self.original_shape[1]
            width = self.original_shape[2]
            img = img.contiguous().view(height, channels, width).transpose(0, 1)
        if self.normalized:
            img = img * 255
        print(img.shape)
        plt.imshow(img.permute(1, 2, 0).int())

    def __len__(self) -> int:
        return self.img.shape[0]

    def __getitem__(self, index) -> Tuple[torch.Tensor]:
        return self.img[index], self.label[index]

    def prepare_dataset(self) -> None:
        """Prepare dataset to be used in model"""
        if self.img is not None:
            self.normalize()
            self.flatten_images()
            self.to(self.device)
