r"""Tests for shared layers: ConvNd, LayerNorm, Patchify, Unpatchify."""

import math
import pytest
import torch

from shaggy.shaggy.layers import ConvNd, LayerNorm, Patchify, Unpatchify


@pytest.mark.parametrize("spatial", [1, 2, 3])
def test_convnd_output_shape(spatial: int) -> None:
    r"""ConvNd produces the expected output shape for 1D, 2D, and 3D inputs."""
    in_channels, out_channels = 4, 8
    spatial_dims = [16] * spatial
    conv = ConvNd(in_channels, out_channels, spatial=spatial, kernel_size=3, padding=1)
    x = torch.randn(2, in_channels, *spatial_dims)
    assert conv(x).shape == (2, out_channels, *spatial_dims)


def test_convnd_unsupported_spatial() -> None:
    r"""ConvNd raises NotImplementedError for spatial dimensions other than 1, 2, 3."""
    with pytest.raises(NotImplementedError):
        ConvNd(4, 8, spatial=4, kernel_size=3, padding=1)


def test_convnd_identity_init() -> None:
    r"""With identity_init, the kernel center slice is close to the identity matrix."""
    conv = ConvNd(4, 4, spatial=2, kernel_size=3, padding=1, identity_init=True)
    # weight shape: (out_channels, in_channels, H, W) -> center slice at [;, :, 1, 1]
    center = conv.weight.data[:, :, 1, 1]
    assert torch.allclose(center, torch.eye(4), atol=1e-1)


# --- LayerNorm ---


def test_layernorm_zero_mean() -> None:
    r"""LayerNorm output has zero mean along the normalized dimension."""
    norm = LayerNorm(dim=-1)
    x = torch.randn(4, 8, 16)
    y = norm(x)
    assert torch.allclose(y.mean(dim=-1), torch.zeros(4, 8), atol=1e-5)


def test_layernorm_unit_variance() -> None:
    r"""LayerNorm output has unit variance along the normalized dimension."""
    norm = LayerNorm(dim=-1)
    x = torch.randn(4, 8, 16)
    y = norm(x)
    assert torch.allclose(y.var(dim=-1), torch.ones(4, 8), atol=1e-4)


def test_layernorm_preserves_shape() -> None:
    r"""LayerNorm output has the same shape as its input."""
    norm = LayerNorm(dim=(-2, -1))
    x = torch.randn(2, 3, 8, 8)
    assert norm(x).shape == x.shape


def test_layernorm_fp16_stability() -> None:
    r"""LayerNorm upcasts to float32 internally to avoid fp16 numerical issues."""
    norm = LayerNorm(dim=-1)
    x = torch.randn(2, 8, 16, dtype=torch.float16)
    y = norm(x)
    assert y.dtype == torch.float16
    assert torch.all(torch.isfinite(y))


# --- Patchify / Unpatchify ---


@pytest.mark.parametrize(
    "spatial_dims, patch_size",
    [
        ((16,), (2,)),
        ((16, 16), (2, 2)),
        ((8, 8, 8), (2, 2, 2)),
    ],
)
def test_patchify_output_shape(spatial_dims: tuple, patch_size: tuple) -> None:
    r"""Patchify reduces each spatial dimension by its patch factor."""
    C = 4
    x = torch.randn(2, C, *spatial_dims)
    patchify = Patchify(patch_size=patch_size)
    y = patchify(x)
    expected_spatial = tuple(s // p for s, p in zip(spatial_dims, patch_size))
    expected_channels = C * torch.prod(torch.tensor(patch_size)).item()
    assert y.shape == (2, expected_channels, *expected_spatial)


@pytest.mark.parametrize(
    "spatial_dims, patch_size",
    [
        ((16,), (2,)),
        ((16, 16), (2, 2)),
        ((8, 8, 8), (2, 2, 2)),
    ],
)
def test_unpatchify_output_shape(spatial_dims: tuple, patch_size: tuple) -> None:
    r"""Unpatchify expands each spatial dimension by its patch factor."""
    C = 4

    reduced_spatial = tuple(s // p for s, p in zip(spatial_dims, patch_size))
    expanded_channels = C * math.prod(patch_size)
    x = torch.randn(2, expanded_channels, *reduced_spatial)
    unpatchify = Unpatchify(patch_size=patch_size)
    y = unpatchify(x)
    assert y.shape == (2, C, *spatial_dims)


@pytest.mark.parametrize("patch_size", [(2,), (2, 2), (2, 2, 2)])
def test_patchify_unpatchify_roundtrip(patch_size: tuple) -> None:
    r"""Unpatchify(Patchify(x)) reconstructs x exactly."""
    C = 4
    spatial_dims = tuple(16 for _ in patch_size)
    x = torch.randn(2, C, *spatial_dims)
    roundtrip = Unpatchify(patch_size=patch_size)(Patchify(patch_size=patch_size)(x))
    assert torch.allclose(roundtrip, x)
