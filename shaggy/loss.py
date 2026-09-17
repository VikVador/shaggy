r"""A collection of losses for Autoencoder training."""

__all__ = [
    "loss_reconstruction",
    "loss_geometry_embedding",
    "loss_continuous_ranked_probability_score",
]

import torch

from albus.metrics import continuous_ranked_probability_score
from torch import Tensor
from typing import Optional


def loss_reconstruction(
    input: Tensor,
    target: Tensor,
    weights: Optional[Tensor] = None,
) -> Tensor:
    r"""Computes the (optionally weighted) mean squared error between input and target.

    Arguments:
        input: Input tensor (B, C, L_1, ..., L_N).
        target: Target tensor (B, C, L_1, ..., L_N).
        weights: Broadcastable Weight tensor.

    Returns:
        loss: Scalar mean squared error.
    """

    # Security
    assert input.shape == target.shape, (
        "ERROR (loss_reconstruction) | Input and target must have the same shape."
    )

    loss = (input - target).square()
    loss = loss * weights if weights is not None else loss

    return loss.mean()


def loss_geometry_embedding(x: Tensor, z: Tensor) -> Tensor:
    r"""Computes the Gromov-Monge Embedding (GME) loss between ambient data and latent codes.

    References:
        | Geometry-preserving encoder/decoder in latent generative models (Lee, 2026)
        | https://arxiv.org/abs/2501.09876

    Arguments:
        x: Ambient data (B, C, L_1, ..., L_N).
        z: Latent codes (B, C*, L_1*, ..., L_N*).

    Returns:
        loss: Scalar GME cost.
    """

    # Security
    batch = len(x)
    assert batch > 1, (
        f"ERROR (loss_geometry_embedding) | Expected at least 2 samples to pair, got {batch}."
    )

    # Distances are computed between flattened samples
    if z.dim() > 2:
        z = z.flatten(1)
    if x.dim() > 2:
        x = x.flatten(1)

    # Compute pairwise squared distances
    dx2 = torch.cdist(x, x).square()
    dz2 = torch.cdist(z, z).square()

    # Compute Gromov-Monge cost, log1p stays accurate for small distances
    cost = (torch.log1p(dz2) - torch.log1p(dx2)).square()

    # Self-distances cost exactly zero, hence the sum over the batch * (batch - 1) pairs
    return cost.sum() / (batch * (batch - 1))


def loss_continuous_ranked_probability_score(
    input: Tensor,
    target: Tensor,
    weights: Optional[Tensor] = None,
) -> Tensor:
    r"""Computes the (optionally weighted) fair CRPS between an ensemble and a target.

    References:
        | Fair scores for ensemble forecasts (Ferro, 2014)
        | https://doi.org/10.1002/qj.2270

    Arguments:
        input: Ensemble tensor (B, E, C, L_1, ..., L_N).
        target: Target tensor (B, C, L_1, ..., L_N).
        weights: Broadcastable Weight tensor.

    Returns:
        loss: Scalar fair CRPS.
    """

    # Security
    assert input.shape[:1] + input.shape[2:] == target.shape, (
        "ERROR (loss_continuous_ranked_probability_score) | Input must be (B, E, *target.shape[1:])"
    )

    dims = " ".join(["B", "E", "C", *(f"L_{i}" for i in range(1, input.dim() - 2))])
    loss = continuous_ranked_probability_score(target, input, dims=dims, ensemble="E", reduce="")
    loss = loss * weights if weights is not None else loss

    return loss.mean()
