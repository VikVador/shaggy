r"""Fusion Autoencoder (FAE)."""

__all__ = [
    "FusionAE",
]

import torch.nn as nn

from collections.abc import Sequence
from torch import Tensor

from shaggy.models.ae import AutoEncoder


class FusionAE(nn.Module):
    r"""Creates a Fusion Autoencoder (FAE) from several (trained) encoder-decoder pairs.

    Arguments:
        encoders: Encoder modules, one per data source.
        decoders: Decoder modules, one per data source.
    """

    def __init__(
        self,
        encoders: Sequence[nn.Module],
        decoders: Sequence[nn.Module],
    ) -> None:
        super().__init__()

        # Security
        assert len(encoders) == len(decoders), (
            f"ERROR (FusionAE) | Expected as many encoders as decoders, "
            f"got {len(encoders)} and {len(decoders)}."
        )

        self.autoencoders = nn.ModuleList([
            AutoEncoder(encoder, decoder) for encoder, decoder in zip(encoders, decoders)
        ])

    def encode(self, xs: Sequence[Tensor]) -> list[Tensor]:
        r"""Encodes each data source into its own latent representation.

        Arguments:
            xs: Input tensors, one per data source.

        Returns:
            zs: Latent codes, one per data source.
        """

        # Security
        n_sources = len(self.autoencoders)
        assert len(xs) == n_sources, (
            f"ERROR (FusionAE) | Expected {n_sources} inputs, got {len(xs)}."
        )

        return [autoencoder.encode(x) for autoencoder, x in zip(self.autoencoders, xs)]

    def decode(self, zs: Sequence[Tensor]) -> list[Tensor]:
        r"""Decodes each latent code back into the ambient space of its data source.

        Arguments:
            zs: Latent codes, one per data source.

        Returns:
            xs: Reconstructed tensors, one per data source.
        """

        # Security
        n_sources = len(self.autoencoders)
        assert len(zs) == n_sources, (
            f"ERROR (FusionAE) | Expected {n_sources} latent codes, got {len(zs)}."
        )

        return [autoencoder.decode(z) for autoencoder, z in zip(self.autoencoders, zs)]

    def forward(self, xs: Sequence[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        r"""Encodes and reconstructs each data source separately.

        Arguments:
            xs: Input tensors, one per data source.

        Returns:
            zs: Latent codes, one per data source.
            ys: Reconstructed tensors, one per data source.
        """

        zs = self.encode(xs)
        ys = self.decode(zs)

        return zs, ys
