r"""Tests for HybridMA."""

import torch

from shaggy.optimizers import HybridMA


def make_mlp() -> torch.nn.Module:
    r"""Builds a small 2-layer MLP used across HybridMA tests."""
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.SiLU(), torch.nn.Linear(8, 4))


def test_hybrid_routes_2d_parameters_to_muon() -> None:
    r"""Weight matrices (ndim == 2) are optimized by the Muon sub-optimizer."""
    mlp = make_mlp()
    optimizer = HybridMA(mlp)

    n_2d = sum(p.ndim == 2 for p in mlp.parameters())
    n_other = sum(p.ndim != 2 for p in mlp.parameters())

    assert sum(len(g["params"]) for g in optimizer.opt_muon.param_groups) == n_2d
    assert sum(len(g["params"]) for g in optimizer.opt_adamw.param_groups) == n_other


def test_hybrid_exclude_names_forces_adamw() -> None:
    r"""Names listed in exclude_names are routed to AdamW even if 2D."""
    mlp = make_mlp()
    optimizer = HybridMA(mlp, exclude_names=["0.weight"])

    excluded_param = dict(mlp.named_parameters())["0.weight"]
    muon_params = [p for g in optimizer.opt_muon.param_groups for p in g["params"]]
    adamw_params = [p for g in optimizer.opt_adamw.param_groups for p in g["params"]]

    assert not any(p is excluded_param for p in muon_params)
    assert any(p is excluded_param for p in adamw_params)


def test_hybrid_step_reduces_loss() -> None:
    r"""A few HybridMA steps reduce the loss of a small MLP regression."""
    torch.manual_seed(0)
    mlp = make_mlp()
    x = torch.randn(16, 4)
    y = torch.randn(16, 4)

    optimizer = HybridMA(mlp, lr=1e-2)

    losses = []
    for _ in range(20):
        loss = (mlp(x) - y).square().mean()
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert losses[-1] < losses[0]


def test_hybrid_zero_grad_clears_both_suboptimizers() -> None:
    r"""zero_grad clears gradients routed to both Muon and AdamW."""
    mlp = make_mlp()
    optimizer = HybridMA(mlp)

    x = torch.randn(2, 4)
    loss = mlp(x).square().mean()
    loss.backward()

    optimizer.zero_grad()

    assert all(p.grad is None for p in mlp.parameters())


def test_hybrid_handles_all_2d_parameters() -> None:
    r"""HybridMA works when every parameter is routed to Muon (empty AdamW group)."""
    torch.manual_seed(0)
    mlp = torch.nn.Sequential(
        torch.nn.Linear(4, 8, bias=False),
        torch.nn.SiLU(),
        torch.nn.Linear(8, 4, bias=False),
    )
    x = torch.randn(16, 4)
    y = torch.randn(16, 4)

    optimizer = HybridMA(mlp, lr=1e-2)
    assert optimizer.opt_adamw is None

    loss_before = mlp(x).sub(y).square().mean()
    for _ in range(10):
        loss = mlp(x).sub(y).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert loss < loss_before


def test_hybrid_state_dict_roundtrip() -> None:
    r"""load_state_dict restores the state of both sub-optimizers."""
    mlp = make_mlp()
    optimizer = HybridMA(mlp)

    x = torch.randn(2, 4)
    loss = mlp(x).square().mean()
    loss.backward()
    optimizer.step()

    state = optimizer.state_dict()

    mlp_copy = make_mlp()
    mlp_copy.load_state_dict(mlp.state_dict())
    new_optimizer = HybridMA(mlp_copy)
    new_optimizer.load_state_dict(state)

    assert set(state.keys()) == {"muon", "adamw"}
    assert len(new_optimizer.opt_muon.state) == len(optimizer.opt_muon.state)
