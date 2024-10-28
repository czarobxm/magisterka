"""Kernel transformations for the Performer."""

from typing import Union
import math

import torch
from torch import nn


def relu_kernel_transformation(
    data: torch.Tensor,
    is_query: bool,
    projection_matrix=Union[None, torch.Tensor],
    numerical_stabilizer: float = 0.000001,
) -> torch.Tensor:
    """Computes features for the ReLU-kernel for Random Features
    from https://arxiv.org/pdf/2009.14794.pdf.

    :param data: input tensor of shape [B, L, Nh, Dh], where: B - batch dimension,
    L - attention dimensions, Nh - heads, Dh - features.
    :param is_query: Indicates whether the input data is a query or a query or key.
    :param projection_matrix: random Gaussian matrix of shape [M, D], where M stands
    for the number of random features and each D x D sub-block has pairwise
    orthogonal rows.
    :param numerical_stabilizer: Small-valued positive constant for numerical stability

    :return: ReLu kernel feature map
    """
    del is_query
    relu = nn.ReLU()
    if projection_matrix is None:
        return nn.functional.relu(data) + numerical_stabilizer
    else:
        ratio = 1.0 / math.sqrt(projection_matrix.shape[0])
        data_dash = ratio * torch.einsum("blhd,md->blhm", data, projection_matrix)
        return relu(data_dash) + numerical_stabilizer


def softmax_kernel_transformation(
    data: torch.Tensor,
    is_query: bool,
    projection_matrix=Union[None, torch.Tensor],
    numerical_stabilizer: float = 0.000001,
) -> torch.Tensor:
    """
    Computes features for the softmax-kernel for Random Features
    from https://arxiv.org/pdf/2009.14794.pdf.

    :param data: input tensor of shape [B, L, Nh, Dh], where: B - batch dimension,
    L - attention dimensions, Nh - heads, Dh - features.
    :param is_query: Indicates whether the input data is a query or a query or key.
    :param projection_matrix: random Gaussian matrix of shape [M, D], where M stands
    for the number of random features and each D x D sub-block has pairwise
    orthogonal rows.
    :param numerical_stabilizer: Small-valued positive constant for numerical stability

    :return: Softmax kernel feature map
    """
    data_normalizer = data.shape[-1] ** -0.25
    data = data_normalizer * data
    ratio = projection_matrix.shape[0] ** -0.5
    data_dash = torch.einsum("blhd,md->blhm", data, projection_matrix)

    diag_data = torch.square(data)
    diag_data = torch.sum(diag_data, dim=-1)  # 확인 필요.
    diag_data = diag_data / 2.0
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
            torch.exp(
                data_dash - diag_data - torch.max(data_dash, dim=-1, keepdims=True).values
            )
            + numerical_stabilizer
        )
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash)) + numerical_stabilizer
        )

    return data_dash
