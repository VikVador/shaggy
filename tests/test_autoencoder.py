r"""Tests for convolutional auto-encoders."""

import pytest
import torch

from pathlib import Path

from shaggy.shaggy.models.cae import create_ConvAE

param_combinations = [
    (1, 3, [16], 2, "softclip", 0.5),
    (1, 4, [8], 2, "softclip2", 0.5),
    (1, 3, [8], 2, "tanh", 0.5),
    (1, 3, [8], 2, "asinh", 0.5),
    (1, 3, [8], 2, None, 0.5),
]


@pytest.mark.parametrize(
    "batch_size, in_channels, hid_channels, latent_channels, saturation, saturation_bound",
    param_combinations,
)
def test_cae(
    tmp_path: Path,
    batch_size: int,
    in_channels: int,
    hid_channels: list,
    latent_channels: int,
    saturation: str,
    saturation_bound: float,
) -> None:
    autoencoder = create_ConvAE(
        in_channels=in_channels,
        hid_channels=hid_channels,
        hid_blocks=[1] * len(hid_channels),
        lat_channels=latent_channels,
        saturation=saturation,
        saturation_bound=saturation_bound,
    )

    N_lat = 64
    N_lon = 64

    x = torch.randn(size=(batch_size, in_channels, N_lat, N_lon))
    z, x_reconstructed = autoencoder(x)

    # Saturation
    if saturation not in [None, "asinh"]:
        assert torch.all(z.abs() <= saturation_bound)

    # Shapes
    c_z, h, w = autoencoder.latent_shape((N_lat, N_lon))
    assert z.shape == (batch_size, c_z, h, w)
    assert x_reconstructed.shape == (batch_size, in_channels, N_lat, N_lon)

    # Gradients
    assert z.requires_grad
    assert x_reconstructed.requires_grad

    loss = x_reconstructed.square().sum()
    loss.backward()

    for p in autoencoder.parameters():
        assert p.grad is not None
        assert torch.all(torch.isfinite(p.grad))

    # Save
    torch.save(autoencoder.state_dict(), tmp_path / "autoencoder_state.pth")

    del autoencoder, loss

    # Load
    autoencoder_copy = create_ConvAE(
        in_channels=in_channels,
        hid_channels=hid_channels,
        hid_blocks=[1] * len(hid_channels),
        lat_channels=latent_channels,
        saturation=saturation,
        saturation_bound=saturation_bound,
    )
    autoencoder_copy.load_state_dict(
        torch.load(tmp_path / "autoencoder_state.pth", weights_only=True)
    )

    autoencoder_copy.eval()

    z_copy, x_reconstructed_copy = autoencoder_copy(x)

    assert torch.allclose(z, z_copy)
    assert torch.allclose(x_reconstructed, x_reconstructed_copy)
