r"""Tests for Shaggy layers: BoundaryConv, FRMSNorm, ResidualBlock."""

import pytest
import torch
import torch.nn as nn

from azula.nn.layers import RMSNorm

from shaggy.layers import BoundaryConv, FRMSNorm, ModulatedSequential, ResidualBlock

CONV = dict(kernel_size=3)


def relative_deviation(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    r"""Computes the RMS of y - x relative to the RMS of x.

    Arguments:
        y: Output tensor.
        x: Reference tensor, same shape as y.

    Returns:
        deviation: Scalar relative deviation.
    """

    return (y - x).pow(2).mean().sqrt() / x.pow(2).mean().sqrt()


# --- BoundaryConv ---


@pytest.mark.parametrize("spatial", [1, 2, 3])
def test_boundaryconv_matches_a_plain_convolution(spatial: int) -> None:
    r"""Determines if zero padding on every axis matches a plain convolution with the same weights."""
    conv = BoundaryConv(4, 6, spatial=spatial, kernel_size=3)
    plain = getattr(nn, f"Conv{spatial}d")(4, 6, kernel_size=3, padding=1)
    plain.load_state_dict(conv.state_dict())
    x = torch.randn(2, 4, *[8] * spatial)

    assert torch.allclose(conv(x), plain(x), atol=1e-6)


@pytest.mark.parametrize(
    "mode, periodic, reflect", [("circular", (0, 1), ()), ("reflect", (), (0, 1))]
)
def test_boundaryconv_matches_a_padded_convolution(
    mode: str, periodic: tuple, reflect: tuple
) -> None:
    r"""Determines if padding every axis periodically or by reflection matches PyTorch's padding modes."""
    conv = BoundaryConv(4, 6, spatial=2, kernel_size=(3, 5), periodic=periodic, reflect=reflect)
    plain = nn.Conv2d(4, 6, kernel_size=(3, 5), padding=(1, 2), padding_mode=mode)
    plain.load_state_dict(conv.state_dict())
    x = torch.randn(2, 4, 8, 12)

    assert torch.allclose(conv(x), plain(x), atol=1e-6)


def test_boundaryconv_is_equivariant_to_rolls_along_periodic_axes() -> None:
    r"""Determines if rolling the input along the periodic axis only rolls the output."""
    conv = BoundaryConv(4, 6, spatial=2, kernel_size=3, periodic=(-1,), reflect=(0,))
    x = torch.randn(2, 4, 8, 12)

    assert torch.allclose(conv(x.roll(5, dims=-1)), conv(x).roll(5, dims=-1), atol=1e-6)
    assert not torch.allclose(conv(x.roll(3, dims=-2)), conv(x).roll(3, dims=-2), atol=1e-3)


@pytest.mark.parametrize("periodic, reflect", [((1,), (1,)), ((-1,), (2,)), ((0, 2), (-1,))])
def test_boundaryconv_rejects_an_axis_both_periodic_and_reflected(
    periodic: tuple, reflect: tuple
) -> None:
    r"""Determines if an axis cannot be periodic and reflected at the same time."""
    with pytest.raises(AssertionError, match="not both"):
        BoundaryConv(4, 6, spatial=3, periodic=periodic, reflect=reflect)


@pytest.mark.parametrize("periodic, reflect", [((2,), ()), ((), (-3,))])
def test_boundaryconv_rejects_axes_out_of_range(periodic: tuple, reflect: tuple) -> None:
    r"""Determines if axes beyond the spatial dimensions are rejected."""
    with pytest.raises(AssertionError, match="Axes must be in"):
        BoundaryConv(4, 6, spatial=2, periodic=periodic, reflect=reflect)


@pytest.mark.parametrize("stride", [1, 2])
def test_boundaryconv_preserves_the_resolution(stride: int) -> None:
    r"""Determines if the output is the input divided by the stride, whatever the boundaries."""
    conv = BoundaryConv(4, 6, spatial=3, kernel_size=3, stride=stride, periodic=(2,), reflect=(1,))
    x = torch.randn(2, 4, 8, 8, 8)

    assert conv(x).shape == (2, 6, *[8 // stride] * 3)


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


@pytest.mark.parametrize("checkpointing", [False, True])
def test_residualblocks_compile_in_a_single_graph(checkpointing: bool) -> None:
    r"""Determines if torch.compile traces a stack of blocks without breaking its graph, checkpointed or not."""
    blocks = ModulatedSequential(*[
        ResidualBlock(8, spatial=2, checkpointing=checkpointing, **CONV) for _ in range(2)
    ])
    x = torch.randn(2, 8, 8, 8)

    # The input does not require gradients, as the data of a training step
    torch._dynamo.reset()
    explanation = torch._dynamo.explain(blocks)(x)

    assert explanation.graph_break_count == 0
    assert explanation.graph_count == 1
