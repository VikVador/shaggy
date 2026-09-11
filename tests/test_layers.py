r"""Tests for Shaggy layers: SwiGLU, ResidualBlock, ResidualGroup, ResidualTrunk."""

import pytest
import torch
import torch.nn as nn

from azula.nn.layers import swiglu

from shaggy.layers import ResidualBlock, ResidualGroup, ResidualTrunk, SwiGLU

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


# --- SwiGLU ---


@pytest.mark.parametrize("spatial", [0, 1, 2, 3])
def test_swiglu_halves_channels(spatial: int) -> None:
    r"""Determines if the channel dimension is halved while the spatial ones are kept."""
    x = torch.randn(2, 6, *[4] * spatial)

    assert SwiGLU(spatial=spatial)(x).shape == (2, 3, *[4] * spatial)


@pytest.mark.parametrize("spatial", [0, 1, 2, 3])
def test_swiglu_matches_azula(spatial: int) -> None:
    r"""Determines if it equals azula's channel-last swiglu applied on the channel axis."""
    x = torch.randn(2, 6, *[4] * spatial)

    reference = swiglu(x.movedim(1, -1)).movedim(-1, 1)

    assert torch.allclose(SwiGLU(spatial=spatial)(x), reference)


def test_swiglu_odd_channels() -> None:
    r"""Determines if an odd number of channels, which cannot be split in two, is rejected."""
    x = torch.randn(2, 5, 4, 4)

    with pytest.raises(RuntimeError):
        SwiGLU(spatial=2)(x)


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
