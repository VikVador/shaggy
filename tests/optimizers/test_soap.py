r"""Tests for the SOAP optimizer."""

import pytest
import torch

from shaggy.optimizers import SOAP
from shaggy.optimizers.soap import adam


def test_soap_step_reduces_loss() -> None:
    r"""A few SOAP steps reduce the loss of a small MLP regression."""
    torch.manual_seed(0)
    mlp = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.SiLU(), torch.nn.Linear(8, 4))
    x = torch.randn(16, 4)
    y = torch.randn(16, 4)

    optimizer = SOAP(mlp.parameters(), lr=1e-2, precondition_frequency=2)

    losses = []
    for _ in range(20):
        loss = (mlp(x) - y).square().mean()
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert losses[-1] < losses[0]


@pytest.mark.parametrize(
    "shape, max_precond_size, expected",
    [
        ((256,), 4096, (256,)),
        ((256, 3, 3), 4096, (256, 9)),
        ((4, 3, 3, 3), 10, (4, 3, 9)),
    ],
)
def test_soap_merge_shape(shape: tuple, max_precond_size: int, expected: tuple) -> None:
    r"""merge_shape folds trailing dimensions without exceeding max_precond_size."""
    assert SOAP.merge_shape(shape, max_precond_size=max_precond_size) == expected


def test_soap_skips_parameters_without_gradient() -> None:
    r"""Parameters left without a gradient are untouched by step."""
    weight = torch.nn.Parameter(torch.randn(4, 4))
    frozen = torch.nn.Parameter(torch.randn(4, 4))
    frozen_before = frozen.detach().clone()

    optimizer = SOAP([weight, frozen], lr=1e-2)
    weight.grad = torch.randn_like(weight)
    optimizer.step()

    assert torch.equal(frozen, frozen_before)


def test_soap_precondition_1d_builds_preconditioner() -> None:
    r"""With precondition_1d=True, a 1D parameter gets a non-trivial preconditioner."""
    bias = torch.nn.Parameter(torch.randn(8))
    optimizer = SOAP([bias], lr=1e-2, precondition_1d=True)
    bias.grad = torch.randn_like(bias)
    optimizer.step()

    state = next(iter(optimizer.state.values()))
    assert state["GG"][0] is not None
    assert state["GG"][0].shape == (8, 8)


def test_adam_update_matches_closed_form() -> None:
    r"""The functional adam() helper matches mean / sqrt(var + eps^2)."""
    mean = torch.tensor([1.0, -2.0])
    var = torch.tensor([4.0, 9.0])
    expected = mean / torch.sqrt(var + 1e-8**2)
    assert torch.allclose(adam(mean.clone(), var.clone(), eps=1e-8), expected)
