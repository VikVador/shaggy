r"""Tests for Shaggy layers: FRMSNorm, ResidualBlock, ResidualGroup, ResidualTrunk."""

import pytest
import torch
import torch.nn as nn

from azula.nn.layers import RMSNorm

from shaggy.layers import FRMSNorm, ResidualBlock, ResidualGroup, ResidualTrunk

CONV = dict(kernel_size=3, padding=1)


def relative_deviation(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    r"""Computes the RMS of y - x relative to the RMS of x.

    Arguments:
        y: Output tensor.
        x: Reference tensor, same shape as y.

    Returns:
        deviation: Scalar relative deviation.
    """

    return (y - x).pow(2).mean().sqrt() / x.pow(2).mean().sqrt()


# --- FRMSNorm ---


@pytest.mark.parametrize("spatial", [1, 2, 3])
@pytest.mark.parametrize("mod_features", [0, 4])
def test_frmsnorm_preserves_shape(spatial: int, mod_features: int) -> None:
    r"""Determines if the output has the shape of the input, modulated or not."""
    norm = FRMSNorm(8, mod_features=mod_features, spatial=spatial)
    x = torch.randn(2, 8, *[6] * spatial)

    assert norm(x, torch.randn(2, 4)).shape == x.shape


@pytest.mark.parametrize("mod", [None, torch.randn(2, 4)])
@pytest.mark.parametrize("mod_features", [None, 0])
def test_frmsnorm_without_modulation(mod: torch.Tensor, mod_features: int) -> None:
    r"""Determines if it reduces to azula's RMSNorm without features, or without a vector."""
    x = torch.randn(2, 8, 6, 6)

    reference = RMSNorm(dim=-3)(x)

    assert FRMSNorm(8, mod_features=mod_features, spatial=2).proj is None
    assert torch.allclose(FRMSNorm(8, mod_features=mod_features, spatial=2)(x, mod), reference)
    assert torch.allclose(FRMSNorm(8, mod_features=4, spatial=2)(x, None), reference)


def test_frmsnorm_matches_gamma_beta() -> None:
    r"""Determines if the output is gamma * rms_norm(x) + beta, as given by the projection."""
    norm = FRMSNorm(8, mod_features=4, spatial=2)
    norm.proj.weight.data.normal_()
    norm.proj.bias.data.normal_()

    x = torch.randn(2, 8, 6, 6)
    mod = torch.randn(2, 4)

    gamma, beta = norm.proj(mod).chunk(2, dim=-1)
    expected = (1 + gamma[..., None, None]) * RMSNorm(dim=-3)(x) + beta[..., None, None]

    assert torch.allclose(norm(x, mod), expected, atol=1e-6)


def test_frmsnorm_is_constant_over_space() -> None:
    r"""Determines if every spatial position is modulated alike, by shifting the input around."""
    norm = FRMSNorm(8, mod_features=4, spatial=2)
    norm.proj.weight.data.normal_()

    x = torch.randn(2, 8, 6, 6)
    mod = torch.randn(2, 4)
    shift = dict(shifts=(2, 3), dims=(-2, -1))

    assert torch.allclose(norm(x.roll(**shift), mod), norm(x, mod).roll(**shift), atol=1e-6)


def test_frmsnorm_starts_near_identity() -> None:
    r"""Determines if the modulation barely changes the normalized features at initialization."""
    norm = FRMSNorm(64, mod_features=32, spatial=2)
    x = torch.randn(4, 64, 16, 16)
    mod = torch.randn(4, 32)

    plain = RMSNorm(dim=-3)(x)

    assert relative_deviation(norm(x, mod), plain) < 0.1


def test_frmsnorm_depends_on_the_modulation() -> None:
    r"""Determines if two different vectors give two different outputs."""
    norm = FRMSNorm(8, mod_features=4, spatial=2)
    norm.proj.weight.data.normal_()

    x = torch.randn(2, 8, 6, 6)

    assert not torch.allclose(norm(x, torch.randn(2, 4)), norm(x, torch.randn(2, 4)))


def test_frmsnorm_gradients() -> None:
    r"""Determines if gradients reach the input, the modulation vector and the projection."""
    norm = FRMSNorm(8, mod_features=4, spatial=2)
    x = torch.randn(2, 8, 6, 6, requires_grad=True)
    mod = torch.randn(2, 4, requires_grad=True)

    norm(x, mod).square().sum().backward()

    for tensor in [x.grad, mod.grad, norm.proj.weight.grad]:
        assert tensor is not None and torch.all(torch.isfinite(tensor))


# --- ResidualBlock ---


@pytest.mark.parametrize("spatial", [1, 2, 3])
@pytest.mark.parametrize("ffn_factor", [1, 2])
def test_residualblock_preserves_shape(spatial: int, ffn_factor: int) -> None:
    r"""Determines if the output has the shape of the input, for 1D, 2D and 3D data."""
    block = ResidualBlock(8, ffn_factor=ffn_factor, spatial=spatial, **CONV)
    x = torch.randn(2, 8, *[6] * spatial)

    assert block(x).shape == x.shape


def test_residualblock_starts_near_identity() -> None:
    r"""Determines if the block barely modifies its input at initialization."""
    block = ResidualBlock(64, spatial=2, **CONV)
    x = torch.randn(4, 64, 16, 16)

    assert relative_deviation(block(x), x) < 0.01


def test_residualblock_skip_connection() -> None:
    r"""Determines if the block reduces exactly to the identity once its FFN is silenced."""
    block = ResidualBlock(8, spatial=2, **CONV)

    block.ffn[-1].weight.data.zero_()
    block.ffn[-1].bias.data.zero_()

    x = torch.randn(2, 8, 8, 8)

    assert torch.allclose(block(x), x)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_residualblock_gradients(checkpointing: bool) -> None:
    r"""Determines if gradients reach the input and every parameter."""
    block = ResidualBlock(8, spatial=2, checkpointing=checkpointing, **CONV)
    x = torch.randn(2, 8, 8, 8, requires_grad=True)

    block(x).square().sum().backward()

    assert x.grad is not None and torch.all(torch.isfinite(x.grad))

    for p in block.parameters():
        assert p.grad is not None and torch.all(torch.isfinite(p.grad))


def test_residualblock_checkpointing_is_transparent() -> None:
    r"""Determines if checkpointing leaves outputs and gradients unchanged, dropout included."""
    block = ResidualBlock(8, spatial=2, dropout=0.5, **CONV)
    block.ffn[-1].weight.data.normal_()

    x = torch.randn(2, 8, 8, 8, requires_grad=True)
    results = []

    for checkpointing in [False, True]:
        block.checkpointing = checkpointing
        torch.manual_seed(0)

        y = block(x)
        (grad,) = torch.autograd.grad(y.sum(), x)

        results.append((y, grad))

    (y_plain, grad_plain), (y_ckpt, grad_ckpt) = results

    assert torch.allclose(y_plain, y_ckpt)
    assert torch.allclose(grad_plain, grad_ckpt)


# --- ResidualGroup ---


def test_residualgroup_preserves_shape() -> None:
    r"""Determines if the group keeps the shape of its input."""
    blocks = [ResidualBlock(8, spatial=2, **CONV) for _ in range(3)]
    group = ResidualGroup(8, blocks, spatial=2, **CONV)
    x = torch.randn(2, 8, 8, 8)

    assert group(x).shape == x.shape


def test_residualgroup_starts_near_identity() -> None:
    r"""Determines if the group barely modifies its input at initialization."""
    blocks = [ResidualBlock(32, spatial=2, **CONV) for _ in range(3)]
    group = ResidualGroup(32, blocks, spatial=2, **CONV)
    x = torch.randn(2, 32, 16, 16)

    assert relative_deviation(group(x), x) < 0.01


def test_residualgroup_long_skip() -> None:
    r"""Determines if the group is the identity once its fuse is silenced, whatever its blocks."""
    blocks = [ResidualBlock(8, spatial=2, **CONV) for _ in range(2)]

    for block in blocks:
        block.ffn[-1].weight.data.normal_()

    group = ResidualGroup(8, blocks, spatial=2, **CONV)
    group.fuse.weight.data.zero_()
    group.fuse.bias.data.zero_()

    x = torch.randn(2, 8, 8, 8)

    assert torch.allclose(group(x), x)


def test_residualgroup_nests() -> None:
    r"""Determines if a group accepts other groups as blocks, i.e. residual-in-residual."""
    inner = [ResidualGroup(8, [ResidualBlock(8, spatial=2, **CONV)], spatial=2, **CONV)]
    outer = ResidualGroup(8, inner, spatial=2, **CONV)
    x = torch.randn(2, 8, 8, 8)

    assert outer(x).shape == x.shape


# --- ResidualTrunk ---


def test_residualtrunk_flat_structure() -> None:
    r"""Determines if num_groups = 0 builds a plain stack of num_blocks blocks, no long skip."""
    trunk = ResidualTrunk(8, num_blocks=3, num_groups=0, spatial=2, **CONV)

    assert isinstance(trunk.blocks, nn.Sequential)
    assert len(trunk.blocks) == 3
    assert all(isinstance(block, ResidualBlock) for block in trunk.blocks)
    assert not any(isinstance(m, ResidualGroup) for m in trunk.modules())


def test_residualtrunk_grouped_structure() -> None:
    r"""Determines if num_groups > 0 nests num_groups groups of num_blocks blocks in one skip."""
    trunk = ResidualTrunk(8, num_blocks=2, num_groups=3, spatial=2, **CONV)

    assert isinstance(trunk.blocks, ResidualGroup)
    assert len(trunk.blocks.blocks) == 3

    for group in trunk.blocks.blocks:
        assert isinstance(group, ResidualGroup)
        assert len(group.blocks) == 2
        assert all(isinstance(block, ResidualBlock) for block in group.blocks)


def test_residualtrunk_blocks_are_independent() -> None:
    r"""Determines if every block is a distinct module, i.e. weights are not shared."""
    trunk = ResidualTrunk(8, num_blocks=2, num_groups=2, spatial=2, **CONV)

    blocks = [m for m in trunk.modules() if isinstance(m, ResidualBlock)]

    assert len(blocks) == 4
    assert len({id(block.ffn[0].weight) for block in blocks}) == 4


@pytest.mark.parametrize("spatial", [1, 2, 3])
@pytest.mark.parametrize("num_groups", [0, 2])
def test_residualtrunk_preserves_shape(spatial: int, num_groups: int) -> None:
    r"""Determines if the trunk keeps the shape of its input, flat or grouped, in 1D, 2D and 3D."""
    trunk = ResidualTrunk(8, num_blocks=2, num_groups=num_groups, spatial=spatial, **CONV)
    x = torch.randn(2, 8, *[6] * spatial)

    assert trunk(x).shape == x.shape


@pytest.mark.parametrize("num_groups", [0, 2])
def test_residualtrunk_starts_near_identity(num_groups: int) -> None:
    r"""Determines if the whole trunk barely modifies its input at initialization."""
    trunk = ResidualTrunk(32, num_blocks=4, num_groups=num_groups, spatial=2, **CONV)
    x = torch.randn(2, 32, 16, 16)

    assert relative_deviation(trunk(x), x) < 0.05


def test_residualtrunk_empty() -> None:
    r"""Determines if a flat trunk without blocks is exactly the identity."""
    trunk = ResidualTrunk(8, num_blocks=0, num_groups=0, spatial=2, **CONV)
    x = torch.randn(2, 8, 8, 8)

    assert torch.equal(trunk(x), x)


@pytest.mark.parametrize("num_groups", [0, 2])
def test_residualtrunk_gradients(num_groups: int) -> None:
    r"""Determines if gradients reach the input and every parameter, flat or grouped."""
    trunk = ResidualTrunk(8, num_blocks=2, num_groups=num_groups, spatial=2, **CONV)
    x = torch.randn(2, 8, 8, 8, requires_grad=True)

    trunk(x).square().sum().backward()

    assert x.grad is not None and torch.all(torch.isfinite(x.grad))

    for p in trunk.parameters():
        assert p.grad is not None and torch.all(torch.isfinite(p.grad))
