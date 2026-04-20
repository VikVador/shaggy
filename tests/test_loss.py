r"""Tests for AELoss."""

import pytest
import torch
import torch.nn as nn

from shaggy.loss import AELoss


def test_aeloss_perfect_reconstruction() -> None:
    r"""Loss is zero when input equals target."""
    loss_fn = AELoss()
    x = torch.randn(2, 3, 8, 8)
    assert loss_fn(x, x).item() == pytest.approx(0.0)


def test_aeloss_known_value() -> None:
    r"""Loss equals 1.0 when the pointwise error is uniformly 1."""
    loss_fn = AELoss()
    input = torch.ones(2, 3, 8, 8)
    target = torch.zeros(2, 3, 8, 8)
    assert loss_fn(input, target).item() == pytest.approx(1.0)


def test_aeloss_no_weights() -> None:
    r"""Without weights, AELoss matches nn.MSELoss exactly."""
    loss_fn = AELoss()
    ref = nn.MSELoss()
    x = torch.randn(2, 3, 8, 8)
    y = torch.randn(2, 3, 8, 8)
    assert loss_fn(x, y).item() == pytest.approx(ref(x, y).item())


def test_aeloss_uniform_weights_invariant() -> None:
    r"""Uniform weights of value c scale the loss by c."""
    x = torch.randn(2, 3, 8, 8)
    y = torch.randn(2, 3, 8, 8)
    ref = AELoss()(x, y).item()
    for c in [0.5, 1.0, 2.0]:
        w = torch.full_like(x, c)
        assert AELoss(weights=w)(x, y).item() == pytest.approx(ref * c, rel=1e-5)


def test_aeloss_weights_zero_channel() -> None:
    r"""Setting weights to zero on a channel excludes it from the loss."""
    x = torch.randn(2, 3, 8, 8)
    y = torch.randn(2, 3, 8, 8)

    w = torch.ones(2, 3, 8, 8)
    w[:, 0, :, :] = 0.0

    expected = (nn.MSELoss(reduction="none")(x, y) * w).mean().item()
    assert AELoss(weights=w)(x, y).item() == pytest.approx(expected, rel=1e-5)


def test_aeloss_weights_move_with_model() -> None:
    r"""Weights stored via register_buffer follow .to(device) calls."""
    w = torch.ones(2, 3, 8, 8)
    loss_fn = AELoss(weights=w)
    loss_fn.cpu()
    assert loss_fn.weights.device.type == "cpu"
