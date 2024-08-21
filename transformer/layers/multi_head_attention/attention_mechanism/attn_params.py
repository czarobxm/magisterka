from typing import Callable

from transformer.layers.multi_head_attention.attention_mechanism.performer.kernel_transformations import (
    softmax_kernel_transformation,
)
from transformer.layers.multi_head_attention.attention_mechanism.performer.utils import (
    orthogonal_gaussian_random_feature,
)


class LinearAttnParams:
    """Base class for linear attention hyperparameters."""

    def __init__(self, method: str):
        self.method = method

    def get_hyperparams(self):
        params = self.__dict__
        return params


class VanillaParams(LinearAttnParams):
    """Vanilla attention hyperparameters."""

    def __init__(self) -> None:
        super().__init__("vanilla")


class PerformerParams(LinearAttnParams):
    """Performer attention hyperparameters."""

    def __init__(
        self,
        kernel_transformation: Callable = softmax_kernel_transformation,
        random_features_num: Callable = 256,
        random_features_gen: Callable = orthogonal_gaussian_random_feature,
    ) -> None:
        super().__init__("performer")
        self.kernel_transformation = kernel_transformation
        self.random_features_num = random_features_num
        self.random_features_gen = random_features_gen


class CosformerParams(LinearAttnParams):
    """Cosformer attention hyperparameters."""

    def __init__(self, eps=1e-6) -> None:
        super().__init__("cosformer")
        self.eps = eps
