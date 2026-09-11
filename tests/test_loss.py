r"""Tests for Shaggy losses: loss_reconstruction, loss_geometry_embedding, loss_crps."""

import pytest
import torch
import torch.nn as nn

from shaggy.loss import (
    loss_continuous_ranked_probability_score,
    loss_geometry_embedding,
    loss_reconstruction,
)

# --- loss_reconstruction ---


def test_loss_reconstruction_perfect() -> None:
    r"""Determines if the loss vanishes when the reconstruction matches the target."""
    x = torch.randn(2, 3, 8, 8)

    assert loss_reconstruction(x, x).item() == pytest.approx(0.0)


def test_loss_reconstruction_matches_mse() -> None:
    r"""Determines if the unweighted loss is exactly the mean squared error."""
    x = torch.randn(2, 3, 8, 8)
    y = torch.randn(2, 3, 8, 8)

    assert loss_reconstruction(x, y).item() == pytest.approx(nn.MSELoss()(x, y).item())


def test_loss_reconstruction_weights_exclude_channel() -> None:
    r"""Determines if a channel weighted by zero no longer contributes to the loss."""
    x = torch.randn(2, 3, 8, 8)
    y = torch.randn(2, 3, 8, 8)

    weights = torch.ones(2, 3, 8, 8)
    weights[:, 0] = 0.0

    y_shifted = y.clone()
    y_shifted[:, 0] += 100.0

    assert loss_reconstruction(x, y, weights).item() == pytest.approx(
        loss_reconstruction(x, y_shifted, weights).item()
    )


def test_loss_reconstruction_weights_broadcast() -> None:
    r"""Determines if a land mask (C, Y, X) broadcasts over the batch dimension."""
    x = torch.randn(4, 3, 8, 8)
    y = torch.randn(4, 3, 8, 8)

    mask = torch.randint(0, 2, (3, 8, 8)).float()

    assert loss_reconstruction(x, y, mask).item() == pytest.approx(
        loss_reconstruction(x, y, mask.expand(4, 3, 8, 8)).item()
    )


def test_loss_reconstruction_shape_mismatch() -> None:
    r"""Determines if mismatched input and target shapes are rejected."""
    x = torch.randn(2, 3, 8, 8)
    y = torch.randn(2, 4, 8, 8)

    with pytest.raises(AssertionError):
        loss_reconstruction(x, y)


def test_loss_reconstruction_gradients() -> None:
    r"""Determines if gradients flow back to the reconstruction."""
    x = torch.randn(2, 3, 8, 8, requires_grad=True)
    y = torch.randn(2, 3, 8, 8)

    loss_reconstruction(x, y, torch.rand(3, 8, 8)).backward()

    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))


# --- loss_geometry_embedding ---


def test_loss_geometry_embedding_identity() -> None:
    r"""Determines if the cost vanishes when latent codes are the ambient data itself."""
    x = torch.randn(5, 6)

    assert loss_geometry_embedding(x, x).item() == pytest.approx(0.0, abs=1e-6)


def test_loss_geometry_embedding_rigid_motion() -> None:
    r"""Determines if the cost vanishes for an isometry, here a rotation and a translation."""
    x = torch.randn(5, 6, dtype=torch.float64)
    rotation, _ = torch.linalg.qr(torch.randn(6, 6, dtype=torch.float64))

    z = x @ rotation + 3.0

    assert loss_geometry_embedding(x, z).item() == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_loss_geometry_embedding_rescaled(scale: float) -> None:
    r"""Determines if the cost is positive when latent distances are rescaled."""
    x = torch.randn(5, 6)

    assert loss_geometry_embedding(x, scale * x).item() > 0.0


def test_loss_geometry_embedding_symmetric() -> None:
    r"""Determines if swapping ambient data and latent codes leaves the cost unchanged."""
    x = torch.randn(5, 6)
    z = torch.randn(5, 3)

    assert loss_geometry_embedding(x, z).item() == pytest.approx(
        loss_geometry_embedding(z, x).item()
    )


def test_loss_geometry_embedding_flattens_inputs() -> None:
    r"""Determines if spatial dimensions are flattened per sample before the distances."""
    x = torch.randn(5, 3, 8, 8)
    z = torch.randn(5, 2, 4, 4)

    assert loss_geometry_embedding(x, z).item() == pytest.approx(
        loss_geometry_embedding(x.flatten(1), z.flatten(1)).item()
    )


def test_loss_geometry_embedding_excludes_self_distances() -> None:
    r"""Determines if the cost averages over off-diagonal pairs only, i.e. B * (B - 1) of them."""
    B = 4
    x = torch.randn(B, 6, dtype=torch.float64)
    z = torch.randn(B, 3, dtype=torch.float64)

    dx2 = torch.cdist(x, x).pow(2)
    dz2 = torch.cdist(z, z).pow(2)
    cost = torch.log((1.0 + dz2) / (1.0 + dx2)).pow(2)

    expected = cost.sum() / (B * (B - 1))

    assert loss_geometry_embedding(x, z).item() == pytest.approx(expected.item())


def test_loss_geometry_embedding_gradients() -> None:
    r"""Determines if gradients flow back to the latent codes."""
    x = torch.randn(5, 6)
    z = torch.randn(5, 3, requires_grad=True)

    loss_geometry_embedding(x, z).backward()

    assert z.grad is not None
    assert torch.all(torch.isfinite(z.grad))


# --- loss_continuous_ranked_probability_score ---


def test_loss_crps_matches_brute_force() -> None:
    r"""Determines if it matches a brute-force double loop over the fair CRPS formula."""
    x = torch.randn(3, 6, 2, 4, dtype=torch.float64)
    y = torch.randn(3, 2, 4, dtype=torch.float64)

    E = x.shape[1]
    mae = (x - y.unsqueeze(1)).abs().mean(dim=1)
    spread = sum((x[:, i] - x[:, j]).abs() for i in range(E) for j in range(E))
    expected = (mae - spread / (2 * E * (E - 1))).mean()

    assert loss_continuous_ranked_probability_score(x, y).item() == pytest.approx(expected.item())


def test_loss_crps_perfect_ensemble() -> None:
    r"""Determines if the loss vanishes when every member equals the target."""
    y = torch.randn(2, 3, 8, 8)
    x = y.unsqueeze(1).expand(2, 5, 3, 8, 8)

    assert loss_continuous_ranked_probability_score(x, y).item() == pytest.approx(0.0, abs=1e-6)


def test_loss_crps_single_member_is_mae() -> None:
    r"""Determines if a single-member (deterministic) ensemble reduces to the MAE."""
    x = torch.randn(2, 1, 3, 8, 8)
    y = torch.randn(2, 3, 8, 8)

    expected = (x.squeeze(1) - y).abs().mean()

    assert loss_continuous_ranked_probability_score(x, y).item() == pytest.approx(expected.item())


def test_loss_crps_rewards_spread() -> None:
    r"""Determines if a spread ensemble scores better than a collapsed one at the same mean."""
    y = torch.zeros(4, 1, 1)
    collapsed = torch.ones(4, 2, 1, 1)
    spread = torch.tensor([0.0, 2.0]).view(1, 2, 1, 1).expand(4, 2, 1, 1)

    assert loss_continuous_ranked_probability_score(
        spread, y
    ) < loss_continuous_ranked_probability_score(collapsed, y)


def test_loss_crps_weights_broadcast() -> None:
    r"""Determines if a land mask (C, Y, X) broadcasts over the batch dimension."""
    x = torch.randn(4, 5, 3, 8, 8)
    y = torch.randn(4, 3, 8, 8)

    mask = torch.randint(0, 2, (3, 8, 8)).float()

    assert loss_continuous_ranked_probability_score(x, y, mask).item() == pytest.approx(
        loss_continuous_ranked_probability_score(x, y, mask.expand(4, 3, 8, 8)).item()
    )


def test_loss_crps_weights_exclude_channel() -> None:
    r"""Determines if a channel weighted by zero no longer contributes to the loss."""
    x = torch.randn(2, 5, 3, 8, 8)
    y = torch.randn(2, 3, 8, 8)

    weights = torch.ones(3, 8, 8)
    weights[0] = 0.0

    y_shifted = y.clone()
    y_shifted[:, 0] += 100.0

    assert loss_continuous_ranked_probability_score(x, y, weights).item() == pytest.approx(
        loss_continuous_ranked_probability_score(x, y_shifted, weights).item()
    )


def test_loss_crps_shape_mismatch() -> None:
    r"""Determines if an input without the ensemble axis is rejected."""
    x = torch.randn(2, 3, 8, 8)
    y = torch.randn(2, 3, 8, 8)

    with pytest.raises(AssertionError):
        loss_continuous_ranked_probability_score(x, y)


def test_loss_crps_gradients() -> None:
    r"""Determines if gradients flow back to every ensemble member."""
    x = torch.randn(2, 5, 3, 8, 8, requires_grad=True)
    y = torch.randn(2, 3, 8, 8)

    loss_continuous_ranked_probability_score(x, y, torch.rand(3, 8, 8)).backward()

    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))
    assert torch.all(x.grad.flatten(2).abs().sum(dim=-1) > 0)
