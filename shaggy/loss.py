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


def loss_reconstruction(input: Tensor, target: Tensor, weights: Optional[Tensor] = None) -> Tensor:
    r"""Computes the (optionally weighted) mean squared error between input and target.

    Arguments:
        input: Input tensor (B, C, L_1, ..., L_N).
        target: Target tensor (B, C, L_1, ..., L_N).
        weights: Broadcastable Weight tensor.

    Returns:
        loss: Scalar mean squared error.
    """

    # Security
    assert (
        input.shape == target.shape
    ), "ERROR (loss_reconstruction) | Input and target must have the same shape."

    loss = torch.pow(input - target, 2)
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
    if z.dim() > 2:
        z = z.flatten(1)
    if x.dim() > 2:
        x = x.flatten(1)

    # Compute pairwise squared distances
    dx2 = torch.pow(torch.cdist(x, x), 2)
    dz2 = torch.pow(torch.cdist(z, z), 2)

    # Compute Gromov-Monge cost
    cost = torch.pow(torch.log((1.0 + dz2) / (1.0 + dx2)), 2)

    # Removing self-distances from the cost matrix
    off_diag = ~torch.eye(cost.shape[0], dtype=torch.bool, device=cost.device)
    return cost[off_diag].mean()


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
    assert (
        input.shape[:1] + input.shape[2:] == target.shape
    ), "ERROR (loss_continuous_ranked_probability_score) | Input must be (B, E, *target.shape[1:])"

    # Pointwise CRPS (B, C, L_1, ..., L_N), no reduction
    dims = " ".join(["B", "E", "C", *(f"L_{i}" for i in range(1, input.dim() - 2))])
    loss = continuous_ranked_probability_score(target, input, dims=dims, ensemble="E", reduce="")

    loss = loss * weights if weights is not None else loss
    return loss.mean()
