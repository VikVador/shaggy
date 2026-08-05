r"""Tests for the Muon optimizer."""

import pytest
import torch

from shaggy.optimizers import Muon


def test_muon_step_reduces_loss() -> None:
    r"""A few Muon steps reduce the loss of a simple regression on a 2D weight."""
    torch.manual_seed(0)
    weight = torch.nn.Parameter(torch.randn(8, 8))
    target = torch.randn(8, 8)

    optimizer = Muon([weight], lr=0.1)

    losses = []
    for _ in range(10):
        loss = (weight - target).square().mean()
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert losses[-1] < losses[0]


def test_muon_rejects_non_2d_parameters() -> None:
    r"""Muon raises a ValueError when given a parameter that is not 2D."""
    bias = torch.nn.Parameter(torch.randn(8))
    with pytest.raises(ValueError):
        Muon([bias])


def test_muon_zero_grad_clears_gradients() -> None:
    r"""zero_grad sets parameter gradients to None."""
    weight = torch.nn.Parameter(torch.randn(4, 4))
    optimizer = Muon([weight])
    weight.grad = torch.randn_like(weight)
    optimizer.zero_grad()
    assert weight.grad is None


def test_muon_state_dict_roundtrip() -> None:
    r"""load_state_dict restores the momentum buffer saved by state_dict."""
    torch.manual_seed(0)
    weight = torch.nn.Parameter(torch.randn(4, 4))
    optimizer = Muon([weight], lr=0.1)
    weight.grad = torch.randn_like(weight)
    optimizer.step()

    state = optimizer.state_dict()

    new_weight = torch.nn.Parameter(weight.detach().clone())
    new_optimizer = Muon([new_weight], lr=0.1)
    new_optimizer.load_state_dict(state)

    buffer = next(iter(optimizer.state.values()))["momentum_buffer"]
    new_buffer = next(iter(new_optimizer.state.values()))["momentum_buffer"]
    assert torch.allclose(buffer, new_buffer)
