r"""Shared Autoencoder base class."""

__all__ = [
    "AutoEncoder",
]

import torch.nn as nn

from azula.nn.utils import get_module_dtype
from torch import Tensor


class AutoEncoder(nn.Module):
    r"""Base class for Autoencoder architectures.

    Arguments:
        encoder: Encoder module.
        decoder: Decoder module.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def latent(self, *args, **kwargs) -> tuple[int, ...]:
        r"""Computes the latent shape."""
        raise NotImplementedError()

    def compression(self, *args, **kwargs) -> tuple[tuple[int, ...], int]:
        r"""Computes the compression factor for a given data shape."""
        raise NotImplementedError()

    def encode(self, x: Tensor) -> Tensor:
        r"""Encodes data in ambient space into a latent representation.

        Arguments:
            x: Input tensor.

        Returns:
            z: Latent code.
        """

        dtype = get_module_dtype(self.encoder)
        z = self.encoder(x.to(dtype))
        return z.to(x.dtype)

    def decode(self, z: Tensor) -> Tensor:
        r"""Decodes a latent code back into ambient space.

        Arguments:
            z: Latent code.

        Returns:
            x: Reconstructed tensor.
        """

        dtype = get_module_dtype(self.decoder)
        x = self.decoder(z.to(dtype))
        return x.to(z.dtype)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        r"""Encodes and reconstructs data.

        Arguments:
            x: Input tensor.

        Returns:
            z: Latent code.
            y: Reconstructed tensor.
        """

        z = self.encode(x)
        y = self.decode(z)

        return z, y
