r"""Tests for convolutional Autoencoders."""

import pytest
import torch

from pathlib import Path

from shaggy.layers import FRMSNorm, ResidualGroup
from shaggy.models.cae import broadcast_to_axes, create_ConvAE

param_combinations = [
    (1, 3, [16], 2),
    (1, 4, [8], 2),
    (2, 3, [8, 16], 2),
]


@pytest.mark.parametrize(
    "batch_size, in_channels, hid_channels, latent_channels",
    param_combinations,
)
def test_cae(
    tmp_path: Path,
    batch_size: int,
    in_channels: int,
    hid_channels: list,
    latent_channels: int,
) -> None:
    r"""Checks forward shapes, gradient flow, and a save/load roundtrip for ConvAE."""
    config = dict(
        in_channels=in_channels,
        out_channels=in_channels,
        hid_channels=hid_channels,
        hid_blocks=[1] * len(hid_channels),
        lat_channels=latent_channels,
        spatial=2,
    )

    autoencoder = create_ConvAE(**config)

    N_lat = 64
    N_lon = 64

    x = torch.randn(size=(batch_size, in_channels, N_lat, N_lon))
    z, x_reconstructed = autoencoder(x)

    # Shapes
    c_z, h, w = autoencoder.latent((N_lat, N_lon))
    assert z.shape == (batch_size, c_z, h, w)
    assert x_reconstructed.shape == (batch_size, in_channels, N_lat, N_lon)

    # Gradients
    loss = x_reconstructed.square().sum()
    loss.backward()

    for p in autoencoder.parameters():
        assert p.grad is not None
        assert torch.all(torch.isfinite(p.grad))

    # Save / load roundtrip
    torch.save(autoencoder.state_dict(), tmp_path / "autoencoder_state.pth")

    autoencoder_copy = create_ConvAE(**config)
    autoencoder_copy.load_state_dict(
        torch.load(tmp_path / "autoencoder_state.pth", weights_only=True)
    )
    autoencoder_copy.eval()

    z_copy, x_reconstructed_copy = autoencoder_copy(x)

    assert torch.allclose(z, z_copy)
    assert torch.allclose(x_reconstructed, x_reconstructed_copy)


def test_cae_compression() -> None:
    r"""Determines if the compression factor accounts for the channel dimension."""
    autoencoder = create_ConvAE(
        in_channels=4,
        out_channels=4,
        hid_channels=[8],
        hid_blocks=[1],
        lat_channels=2,
        spatial=2,
    )

    latent, factor = autoencoder.compression((4, 32, 32))

    assert latent == (2, 32, 32)
    assert factor == 2


def test_cae_asymmetric() -> None:
    r"""Determines if per-side configurations build an asymmetric autoencoder."""
    autoencoder = create_ConvAE(
        in_channels=3,
        out_channels=3,
        lat_channels=4,
        spatial=2,
        config_encoder=dict(hid_channels=[8, 16], hid_blocks=[1, 1]),
        config_decoder=dict(hid_channels=[32, 64], hid_blocks=[3, 3], ffn_factor=2),
    )

    n_encoder = sum(p.numel() for p in autoencoder.encoder.parameters())
    n_decoder = sum(p.numel() for p in autoencoder.decoder.parameters())

    assert n_decoder > 10 * n_encoder

    # Both sides resample by the same factor, so the reconstruction matches the input
    x = torch.randn(2, 3, 32, 32)
    _, x_reconstructed = autoencoder(x)

    assert x_reconstructed.shape == x.shape


def test_cae_asymmetric_rejects_mismatched_scales() -> None:
    r"""A decoder that does not undo the encoder's downsampling is rejected at build time."""
    with pytest.raises(AssertionError, match="downsamples"):
        create_ConvAE(
            in_channels=3,
            out_channels=3,
            lat_channels=4,
            spatial=2,
            config_encoder=dict(hid_channels=[8, 16, 32], hid_blocks=[1, 1, 1]),
            config_decoder=dict(hid_channels=[8, 16], hid_blocks=[1, 1]),
        )


def test_cae_residual_in_residual() -> None:
    r"""Determines if hid_groups turns each depth into a residual-in-residual trunk."""
    shared = dict(
        in_channels=3,
        out_channels=3,
        lat_channels=4,
        spatial=2,
        hid_channels=[8, 16],
        hid_blocks=[2, 2],
    )

    flat = create_ConvAE(**shared)
    nested = create_ConvAE(**shared, hid_groups=[2, 2])

    # The nested trunks add one fusing convolution per group, plus one per depth
    assert any(isinstance(m, ResidualGroup) for m in nested.encoder.modules())
    assert not any(isinstance(m, ResidualGroup) for m in flat.encoder.modules())

    n_flat = sum(p.numel() for p in nested.parameters())
    assert n_flat > sum(p.numel() for p in flat.parameters())

    # Same latent shape and same reconstruction shape as the flat trunk
    x = torch.randn(2, 3, 32, 32)
    z_flat, _ = flat(x)
    z_nested, x_reconstructed = nested(x)

    assert z_flat.shape == z_nested.shape
    assert x_reconstructed.shape == x.shape

    x_reconstructed.square().sum().backward()

    for p in nested.parameters():
        assert p.grad is not None
        assert torch.all(torch.isfinite(p.grad))


# --- broadcast_to_axes ---


@pytest.mark.parametrize("spatial", [1, 2, 3])
def test_broadcast_to_axes_integer(spatial: int) -> None:
    r"""Determines if a single integer is repeated once per spatial axis."""
    assert broadcast_to_axes(3, spatial) == (3,) * spatial


def test_broadcast_to_axes_sequence() -> None:
    r"""Determines if a sequence with one value per axis is returned as a tuple, order kept."""
    assert broadcast_to_axes([2, 4, 1], spatial=3) == (2, 4, 1)


@pytest.mark.parametrize("value", [(2, 2), (2, 2, 2, 2)])
def test_broadcast_to_axes_wrong_length(value: tuple) -> None:
    r"""Determines if a sequence whose length differs from the number of axes is rejected."""
    with pytest.raises(AssertionError):
        broadcast_to_axes(value, spatial=3)


def test_broadcast_to_axes_rejected_by_encoder() -> None:
    r"""Determines if a mismatched kernel size fails when building the model, not later."""
    with pytest.raises(AssertionError):
        create_ConvAE(2, 2, 4, spatial=3, kernel_size=(3, 3), hid_channels=[8], hid_blocks=[1])


# --- ConvEncoder / ConvDecoder security ---


@pytest.mark.parametrize("kernel_size", [2, (3, 4)])
def test_cae_rejects_even_kernel(kernel_size: object) -> None:
    r"""Determines if even kernel sizes, which would change the resolution, are rejected."""
    with pytest.raises(AssertionError, match="odd"):
        create_ConvAE(2, 2, 4, kernel_size=kernel_size, hid_channels=[8], hid_blocks=[1])


def test_cae_rejects_mismatched_depths() -> None:
    r"""Determines if hid_channels and hid_blocks of different lengths are rejected."""
    with pytest.raises(AssertionError, match="match in length"):
        create_ConvAE(2, 2, 4, hid_channels=[8, 16], hid_blocks=[1])


# --- Modulation ---

MOD_CONFIG = dict(
    in_channels=2,
    out_channels=2,
    lat_channels=4,
    hid_channels=[8, 16],
    hid_blocks=[1, 1],
    hid_groups=[1, 1],
    spatial=2,
)


def modulated_norms(model: torch.nn.Module) -> list:
    r"""Collects the normalization layers that carry a modulation projection."""
    return [m for m in model.modules() if isinstance(m, FRMSNorm) and m.proj is not None]


def test_cae_without_mod_features_has_no_projection() -> None:
    r"""Determines if mod_features = 0 adds no modulation layer, hence no parameter."""
    model = create_ConvAE(**MOD_CONFIG)

    assert modulated_norms(model) == []
    assert not any(name.endswith("norm.proj.weight") for name, _ in model.named_parameters())


def test_cae_modulation_reaches_every_norm() -> None:
    r"""Determines if the vector reaches every norm of the decoder, and none of the encoder."""
    model = create_ConvAE(**MOD_CONFIG, config_decoder={"mod_features": 4})
    x = torch.randn(2, 2, 16, 16)

    _, y = model(x, torch.randn(2, 4))
    y.square().sum().backward()

    assert modulated_norms(model.encoder) == []
    assert len(modulated_norms(model.decoder)) > 0

    for norm in modulated_norms(model.decoder):
        assert norm.proj.weight.grad is not None
        assert torch.any(norm.proj.weight.grad != 0)


def test_cae_modulation_changes_the_reconstruction() -> None:
    r"""Determines if two different vectors give two different reconstructions."""
    model = create_ConvAE(**MOD_CONFIG, config_decoder={"mod_features": 4})

    for norm in modulated_norms(model):
        norm.proj.weight.data.normal_()

    x = torch.randn(2, 2, 16, 16)

    _, y1 = model(x, torch.randn(2, 4))
    _, y2 = model(x, torch.randn(2, 4))

    assert not torch.allclose(y1, y2)


def test_cae_without_vector_is_deterministic() -> None:
    r"""Determines if a modulated model without a vector behaves like an unmodulated one."""
    model = create_ConvAE(**MOD_CONFIG, config_decoder={"mod_features": 4})

    for norm in modulated_norms(model):
        norm.proj.weight.data.normal_()

    x = torch.randn(2, 2, 16, 16)

    assert torch.allclose(model(x)[1], model(x, None)[1])


def test_cae_modulation_starts_near_identity() -> None:
    r"""Determines if the modulation barely changes the reconstruction at initialization."""
    model = create_ConvAE(**MOD_CONFIG, config_decoder={"mod_features": 4})
    x = torch.randn(2, 2, 16, 16)

    _, plain = model(x)
    _, modulated = model(x, torch.randn(2, 4))

    assert (modulated - plain).pow(2).mean().sqrt() < 0.1 * plain.pow(2).mean().sqrt()


@pytest.mark.parametrize("spatial", [1, 2, 3])
def test_cae_modulation_preserves_shapes(spatial: int) -> None:
    r"""Determines if a modulated autoencoder still reconstructs its input shape, in 1D to 3D."""
    model = create_ConvAE(**{**MOD_CONFIG, "spatial": spatial}, config_decoder={"mod_features": 4})
    x = torch.randn(2, 2, *[8] * spatial)

    _, y = model(x, torch.randn(2, 4))

    assert y.shape == x.shape
