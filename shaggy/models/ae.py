r"""Shared auto-encoder base class."""

__all__ = [
    "AutoEncoder",
]

import torch
import torch.nn as nn

from azula.nn.utils import get_module_dtype
from torch import Tensor
from typing import Optional, Sequence, Tuple


class AutoEncoder(nn.Module):
    r"""Base class shared by all auto-encoder architectures (e.g. ConvAE, GraphAE).

    Subclasses provide an encoder/decoder pair and must implement `latent_shape`
    and `compression_info`; their signatures differ across architectures (e.g. a
    ConvAE's latent shape depends on the input resolution, while a GraphAE's mesh
    already fixes it), so they are not defined here.

    Arguments:
        encoder: Encoder module.
        decoder: Decoder module.
        saturation: Saturation function applied to latent codes.
        saturation_bound: Bound used by the saturation function.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        saturation: Optional[str] = "softclip2",
        saturation_bound: float = 5.0,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.saturation = saturation
        self.saturation_bound = saturation_bound

    def saturate(self, x: Tensor) -> Tensor:
        r"""Applies the configured saturation function to a tensor.

        Arguments:
            x: Input tensor.

        Returns:
            Saturated tensor, with the same shape as x.
        """

        if self.saturation is None:
            return x
        elif self.saturation == "softclip":
            return x / (1 + abs(x) / self.saturation_bound)
        elif self.saturation == "softclip2":
            return x * torch.rsqrt(1 + torch.square(x / self.saturation_bound))
        elif self.saturation == "tanh":
            return torch.tanh(x / self.saturation_bound) * self.saturation_bound
        elif self.saturation == "asinh":
            return torch.arcsinh(x)
        else:
            raise ValueError(f"unknown saturation '{self.saturation}'")

    def latent_shape(self, *args, **kwargs) -> Tuple[int, ...]:
        r"""Returns the latent tensor shape.

        Must be implemented by subclasses: the signature depends on the
        architecture (e.g. ConvAE needs the input resolution, GraphAE's mesh
        already fixes it).
        """
        raise NotImplementedError()

    def compression_info(self, input_shape: Sequence[int]) -> Tuple[Tuple[int, ...], int]:
        r"""Returns the latent shape and compression factor for a given input shape.

        Must be implemented by subclasses, see `latent_shape`.
        """
        raise NotImplementedError()

    def encode(self, x: Tensor) -> Tensor:
        r"""Encodes an input tensor into a latent representation.

        Arguments:
            x: Input tensor.

        Returns:
            z: Latent code.
        """

        dtype = get_module_dtype(self.encoder)
        z = self.encoder(x.to(dtype))
        z = self.saturate(z)

        return z.to(x.dtype)

    def decode(self, z: Tensor) -> Tensor:
        r"""Decodes a latent code back into an output tensor.

        Arguments:
            z: Latent code.

        Returns:
            x: Reconstructed tensor.
        """

        dtype = get_module_dtype(self.decoder)
        x = self.decoder(z.to(dtype))

        return x.to(z.dtype)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Encodes and reconstructs an input tensor.

        Arguments:
            x: Input tensor.

        Returns:
            z: Latent code.
            y: Reconstructed tensor.
        """

        z = self.encode(x)
        y = self.decode(z)

        return z, y
