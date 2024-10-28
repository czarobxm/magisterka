import torch


def iid_gaussian(m: int, d: int) -> torch.Tensor:
    """Generate IID Gaussian random features"""
    return torch.normal(0.0, 1.0, size=(m, d))


def noncausal_numerator(
    qs: torch.Tensor, ks: torch.Tensor, vs: torch.Tensor
) -> torch.Tensor:
    """Computes not-normalized FAVOR noncausal attention AV.

    :param qs: query_prime tensor of the shape [L,B,Nh,M].
    :param ks: key_prime tensor of the shape [L,B,Nh,M].
    :param vs: value tensor of the shape [L,B,Nh,Dh].

    :return: Not-normalized FAVOR noncausal attention AV of shape [L, B, Nh, Dh].
    """
    kvs = torch.einsum("lbhm,lbhd->bhmd", ks, vs)
    return torch.einsum("lbhm,bhmd->lbhd", qs, kvs)


def noncausal_denominator(
    qs: torch.Tensor, ks: torch.Tensor, device: str
) -> torch.Tensor:
    """Computes FAVOR normalizer in noncausal attention.

    :param qs: query_prime tensor of the shape [L,B,H,M].
    :param ks: key_prime tensor of the shape [L,B,H,M].

    :return: FAVOR normalizer in noncausal attention of shape [].
    """
    all_ones = torch.ones([ks.shape[0]]).to(device)
    ks_sum = torch.einsum("lbhm,l->bhm", ks, all_ones)
    return torch.einsum("lbhm,bhm->lbh", qs, ks_sum)


def causal_numerator(
    qs: torch.Tensor, ks: torch.Tensor, vs: torch.Tensor, device: str
) -> torch.Tensor:
    """Computes not-normalized FAVOR causal attention A_{masked}V.

    :param qs: query_prime tensor of the shape [L,B,Nh,M].
    :param ks: key_prime tensor of the shape [L,B,Nh,M].
    :param vs: value tensor of the shape [L,B,Nh,Dh].

    :return: Not-normalized FAVOR causal attention A_{masked}V of shape [L, B, Nh, Dh].
    """

    result = []
    sums = (
        torch.zeros_like(torch.einsum("ijk,ijl->ijkl", ks[0], vs[0]))
        .to(device)
        .to(qs.dtype)
    )

    for index in range(qs.shape[0]):
        sums = sums + torch.einsum("ijk,ijl->ijkl", ks[index], vs[index])
        result.append(torch.einsum("ijkl,ijk->ijl", sums, qs[index])[None, Ellipsis])

    result = torch.cat(result, dim=0)

    return result


def causal_denominator(qs: torch.Tensor, ks: torch.Tensor, device: str) -> torch.Tensor:
    """Computes FAVOR normalizer in causal attention.

    :param qs: query_prime tensor of the shape [L,B,H,M].
    :param ks: key_prime tensor of the shape [L,B,H,M].

    :return: FAVOR normalizer in causal attention of shape [L, B, Nh].
    """

    result = []
    sums = torch.zeros_like(ks[0]).to(device)

    for index in range(qs.shape[0]):
        sums = sums + ks[index]
        result.append(torch.sum(qs[index] * sums, dim=2)[None, Ellipsis])

    result = torch.cat(result, dim=0)

    return result


def orthogonal_square(d: int) -> torch.Tensor:
    """
    Create orthogonal square matrix using Gram-Schmidt

    :param d: dimension of the matrix

    :return: orthogonal square matrix
    """
    q, _ = torch.linalg.qr(iid_gaussian(d, d))  # pylint: disable=not-callable
    return q.T


def orthogonal_gaussian_random_feature(m: int, d: int) -> torch.Tensor:
    """
    Generate orthogonal Gaussian random features

    :param m: number of random features
    :param d: dimension of the input

    :return: orthogonal Gaussian random features
    """

    num_squares = int(m / d)
    blocks = [orthogonal_square(d) for _ in range(num_squares)]

    remainder = m - d * num_squares
    if remainder:
        blocks.append(orthogonal_square(d)[:remainder])

    matrix = torch.cat(blocks)
    matrix /= torch.sqrt(torch.tensor(num_squares + remainder / d))
    # matrix = np.diag(np.sqrt(d) * np.ones(m)) @ matrix

    return matrix
