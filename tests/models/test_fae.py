r"""Tests for Fusion Autoencoders."""

import pytest
import torch

from shaggy.models.cae import create_ConvAE
from shaggy.models.fae import FusionAE


def make_sources() -> tuple:
    r"""Builds a 2D and a 3D ConvAE, with one matching input each."""
    plane = create_ConvAE(2, 2, 4, spatial=2, hid_channels=[8, 16], hid_blocks=[1, 1])
    volume = create_ConvAE(3, 3, 2, spatial=3, hid_channels=[8], hid_blocks=[1])

    xs = [torch.randn(2, 2, 16, 32), torch.randn(2, 3, 8, 16, 4)]

    return [plane, volume], xs


def test_fae_typical() -> None:
    r"""Determines if data of different dimensions yields one latent and reconstruction each."""
    autoencoders, xs = make_sources()
    fae = FusionAE(
        encoders=[ae.encoder for ae in autoencoders],
        decoders=[ae.decoder for ae in autoencoders],
    )

    zs, ys = fae(xs)

    assert len(zs) == len(ys) == 2
    assert zs[0].shape == (2, *autoencoders[0].latent((16, 32)))
    assert zs[1].shape == (2, *autoencoders[1].latent((8, 16, 4)))

    for x, y in zip(xs, ys):
        assert y.shape == x.shape


def test_fae_matches_individual_autoencoders() -> None:
    r"""Determines if each latent code is exactly the one of its own autoencoder."""
    autoencoders, xs = make_sources()
    fae = FusionAE(
        encoders=[ae.encoder for ae in autoencoders],
        decoders=[ae.decoder for ae in autoencoders],
    )

    zs, ys = fae(xs)

    for autoencoder, x, z, y in zip(autoencoders, xs, zs, ys):
        z_ref, y_ref = autoencoder(x)

        assert torch.equal(z, z_ref)
        assert torch.equal(y, y_ref)


def test_fae_gradients() -> None:
    r"""Determines if gradients reach every parameter of every encoder-decoder pair."""
    autoencoders, xs = make_sources()
    fae = FusionAE(
        encoders=[ae.encoder for ae in autoencoders],
        decoders=[ae.decoder for ae in autoencoders],
    )

    _, ys = fae(xs)
    sum(y.square().sum() for y in ys).backward()

    for p in fae.parameters():
        assert p.grad is not None
        assert torch.all(torch.isfinite(p.grad))


def test_fae_rejects_mismatched_pairs() -> None:
    r"""Determines if a different number of encoders and decoders is rejected at build time."""
    autoencoders, _ = make_sources()

    with pytest.raises(AssertionError, match="as many encoders as decoders"):
        FusionAE(
            encoders=[ae.encoder for ae in autoencoders],
            decoders=[autoencoders[0].decoder],
        )


def test_fae_rejects_wrong_number_of_inputs() -> None:
    r"""Determines if a number of inputs different from the number of pairs is rejected."""
    autoencoders, xs = make_sources()
    fae = FusionAE(
        encoders=[ae.encoder for ae in autoencoders],
        decoders=[ae.decoder for ae in autoencoders],
    )

    with pytest.raises(AssertionError, match="inputs"):
        fae(xs[:1])
