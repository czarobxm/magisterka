from typing import Callable, Dict, Any
from dataclasses import dataclass, field

from transformer.multi_head_attention.attention_mechanism.performer.kernel_transformations import (
    softmax_kernel_transformation,
)
from transformer.multi_head_attention.attention_mechanism.performer.utils import (
    orthogonal_gaussian_random_feature,
)


@dataclass
class LinearAttnParams:
    """Base class for linear attention hyperparameters."""

    method: str

    def get_hyperparams(self) -> Dict[str, Any]:
        """Return all hyperparameters as a dictionary."""
        return self.__dict__


@dataclass
class VanillaParams(LinearAttnParams):
    """Vanilla attention hyperparameters."""

    method: str = field(default="vanilla", init=False)


@dataclass
class PerformerParams(LinearAttnParams):
    """Performer attention hyperparameters."""

    method: str = field(default="performer", init=False)
    kernel_transformation: Callable = field(default=softmax_kernel_transformation)
    random_features_num: int = field(default=256)
    random_features_gen: Callable = field(default=orthogonal_gaussian_random_feature)


@dataclass
class CosformerParams(LinearAttnParams):
    """Cosformer attention hyperparameters."""

    method: str = field(default="cosformer", init=False)
    eps: float = field(default=1e-6)
