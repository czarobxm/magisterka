from transformer.layers.multi_head_attention.attention_mechanism.attn_params import (
    CosformerParams,
    PerformerParams,
    VanillaParams,
)
from transformer.layers.multi_head_attention.attention_mechanism.performer.kernel_transformations import (
    softmax_kernel_transformation,
)
from transformer.layers.multi_head_attention.attention_mechanism.performer.utils import (
    orthogonal_gaussian_random_feature,
)


def get_attention_params(args):
    if args.mha_type == "vanilla":
        return VanillaParams()
    elif args.mha_type == "performer":
        kernel_transformation = softmax_kernel_transformation
        random_features_gen = orthogonal_gaussian_random_feature
        return PerformerParams(
            kernel_transformation=kernel_transformation,
            random_features_num=args.random_features_num,
            random_features_gen=random_features_gen,
        )
    elif args.mha_type == "cosformer":
        return CosformerParams()
    else:
        raise NotImplementedError(f"{args.mha_type} attention is not implemented.")
