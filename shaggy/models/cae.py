r"""Convolutional Auto-Encoder (CAE) building blocks."""

__all__ = [
    "ConvEncoder",
    "ConvDecoder",
    "ConvAE",
    "create_ConvAE",
]

import math
import torch
import torch.nn as nn

from azula.nn.utils import get_module_dtype
from torch import Tensor
from typing import Optional, Sequence, Tuple, Union

from shaggy.layers import (
    ConvNd,
    LayerNorm,
    Patchify,
    Unpatchify,
)
from shaggy.utils import checkpoint


class Residual(nn.Sequential):
    r"""Wraps a sequential module with a residual (skip) connection."""

    def forward(self, x: Tensor) -> Tensor:
        return x + super().forward(x)


class ResBlock(nn.Module):
    r"""Creates a residual block module.

    Arguments:
        channels: Number of channels C.
        ffn_factor: Channel expansion factor in the FFN.
        spatial: Number of spatial dimensions N.
        dropout: Dropout rate in [0, 1].
        checkpointing: Whether to use gradient checkpointing or not.
        kwargs: Keyword arguments passed to torch.nn.Conv2d.
    """

    def __init__(
        self,
        channels: int,
        ffn_factor: int = 1,
        spatial: int = 2,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.checkpointing = checkpointing

        # Norm
        self.norm = LayerNorm(dim=-spatial - 1)

        # FFN
        self.ffn = nn.Sequential(
            ConvNd(channels, ffn_factor * channels, spatial=spatial, **kwargs),
            nn.SiLU(),
            nn.Identity() if dropout is None else nn.Dropout(dropout),
            ConvNd(ffn_factor * channels, channels, spatial=spatial, **kwargs),
        )

        self.ffn[-1].weight.data.mul_(1e-2)

    def _forward(self, x: Tensor) -> Tensor:
        r"""Applies layer norm, FFN, and residual addition.

        Arguments:
            x: Input tensor, with shape (B, C, L_1, ..., L_N).

        Returns:
            Output tensor, with shape (B, C, L_1, ..., L_N).
        """

        y = self.norm(x)
        y = self.ffn(y)

        return x + y

    def forward(self, x: Tensor) -> Tensor:
        if self.checkpointing:
            return checkpoint(self._forward, reentrant=not self.training)(x)
        else:
            return self._forward(x)


class ConvEncoder(nn.Module):
    r"""Creates a convolutional encoder.

    Arguments:
        in_channels: Number of input channels C_i.
        out_channels: Number of output channels C_o.
        hid_channels: Numbers of channels at each depth.
        hid_blocks: Numbers of hidden blocks at each depth.
        kernel_size: Kernel size of all convolutions.
        stride: Stride of the downsampling convolutions.
        pixel_shuffle: Whether to use pixel shuffling or not.
        ffn_factor: Channel expansion factor in each FFN.
        spatial: Number of spatial dimensions N.
        patch_size: Patch size applied before the first convolution.
        periodic: Whether the spatial dimensions are periodic or not.
        dropout: Dropout rate in [0, 1].
        checkpointing: Whether to use gradient checkpointing or not.
        identity_init: Initialize down/upsampling convolutions as identity.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hid_channels: Sequence[int] = (64, 128, 256),
        hid_blocks: Sequence[int] = (3, 3, 3),
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 2,
        pixel_shuffle: bool = True,
        ffn_factor: int = 1,
        spatial: int = 2,
        patch_size: Union[int, Sequence[int]] = 1,
        periodic: bool = False,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
        identity_init: bool = True,
    ) -> None:
        super().__init__()

        assert len(hid_blocks) == len(hid_channels)

        self.in_channels = in_channels

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * spatial

        if isinstance(stride, int):
            stride = [stride] * spatial

        if isinstance(patch_size, int):
            patch_size = [patch_size] * spatial

        kwargs = dict(
            kernel_size=tuple(kernel_size),
            padding=tuple(k // 2 for k in kernel_size),
            padding_mode="circular" if periodic else "zeros",
        )

        self.patch = Patchify(patch_size=patch_size)
        self.descent = nn.ModuleList()

        for i, num_blocks in enumerate(hid_blocks):
            blocks = nn.ModuleList()

            if i > 0:
                if pixel_shuffle:
                    blocks.append(
                        nn.Sequential(
                            Patchify(patch_size=stride),
                            ConvNd(
                                hid_channels[i - 1] * math.prod(stride),
                                hid_channels[i],
                                spatial=spatial,
                                identity_init=identity_init,
                                **kwargs,
                            ),
                        )
                    )
                else:
                    blocks.append(
                        ConvNd(
                            hid_channels[i - 1],
                            hid_channels[i],
                            spatial=spatial,
                            stride=stride,
                            identity_init=identity_init,
                            **kwargs,
                        )
                    )
            else:
                blocks.append(
                    ConvNd(
                        math.prod(patch_size) * in_channels,
                        hid_channels[i],
                        spatial=spatial,
                        **kwargs,
                    )
                )

            for _ in range(num_blocks):
                blocks.append(
                    ResBlock(
                        hid_channels[i],
                        ffn_factor=ffn_factor,
                        spatial=spatial,
                        dropout=dropout,
                        checkpointing=checkpointing,
                        **kwargs,
                    )
                )

            if i + 1 == len(hid_blocks):
                blocks.append(
                    ConvNd(
                        hid_channels[i],
                        out_channels,
                        spatial=spatial,
                        identity_init=identity_init,
                        **kwargs,
                    )
                )

            self.descent.append(blocks)

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: Input tensor, with shape (B, C_i, L_1, ..., L_N).

        Returns:
            Output tensor, with shape (B, C_o, L_1 / 2^D, ..., L_N / 2^D).
        """

        x = self.patch(x)

        for blocks in self.descent:
            for block in blocks:
                x = block(x)

        return x


class ConvDecoder(nn.Module):
    r"""Creates a convolutional decoder module.

    Arguments:
        in_channels: Number of input channels C_i.
        out_channels: Number of output channels C_o.
        hid_channels: Numbers of channels at each depth.
        hid_blocks: Numbers of hidden blocks at each depth.
        kernel_size: Kernel size of all convolutions.
        stride: Stride of the downsampling convolutions.
        pixel_shuffle: Whether to use pixel shuffling or not.
        ffn_factor: Channel expansion factor in each FFN.
        spatial: Number of spatial dimensions N.
        patch_size: Patch size applied after the last convolution.
        periodic: Whether the spatial dimensions are periodic or not.
        dropout: Dropout rate in [0, 1].
        checkpointing: Whether to use gradient checkpointing or not.
        identity_init: Initialize down/upsampling convolutions as identity.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hid_channels: Sequence[int] = (64, 128, 256),
        hid_blocks: Sequence[int] = (3, 3, 3),
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 2,
        pixel_shuffle: bool = True,
        ffn_factor: int = 1,
        spatial: int = 2,
        patch_size: Union[int, Sequence[int]] = 1,
        periodic: bool = False,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
        identity_init: bool = True,
    ) -> None:
        super().__init__()

        assert len(hid_blocks) == len(hid_channels)

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * spatial

        if isinstance(stride, int):
            stride = [stride] * spatial

        if isinstance(patch_size, int):
            patch_size = [patch_size] * spatial

        kwargs = dict(
            kernel_size=tuple(kernel_size),
            padding=tuple(k // 2 for k in kernel_size),
            padding_mode="circular" if periodic else "zeros",
        )

        self.unpatch = Unpatchify(patch_size=patch_size)
        self.ascent = nn.ModuleList()

        for i, num_blocks in reversed(list(enumerate(hid_blocks))):
            blocks = nn.ModuleList()

            if i + 1 == len(hid_blocks):
                blocks.append(
                    ConvNd(
                        in_channels,
                        hid_channels[i],
                        spatial=spatial,
                        identity_init=identity_init,
                        **kwargs,
                    )
                )

            for _ in range(num_blocks):
                blocks.append(
                    ResBlock(
                        hid_channels[i],
                        ffn_factor=ffn_factor,
                        spatial=spatial,
                        dropout=dropout,
                        checkpointing=checkpointing,
                        **kwargs,
                    )
                )

            if i > 0:
                if pixel_shuffle:
                    blocks.append(
                        nn.Sequential(
                            ConvNd(
                                hid_channels[i],
                                hid_channels[i - 1] * math.prod(stride),
                                spatial=spatial,
                                identity_init=identity_init,
                                **kwargs,
                            ),
                            Unpatchify(patch_size=stride),
                        )
                    )
                else:
                    blocks.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=tuple(stride), mode="nearest"),
                            ConvNd(
                                hid_channels[i],
                                hid_channels[i - 1],
                                spatial=spatial,
                                identity_init=identity_init,
                                **kwargs,
                            ),
                        )
                    )
            else:
                blocks.append(
                    ConvNd(
                        hid_channels[i],
                        math.prod(patch_size) * out_channels,
                        spatial=spatial,
                        **kwargs,
                    )
                )

            self.ascent.append(blocks)

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: Input tensor, with shape (B, C_i, L_1, ..., L_N).

        Returns:
            Output tensor, with shape (B, C_o, L_1 * 2^D, ..., L_N * 2^D).
        """

        for blocks in self.ascent:
            for block in blocks:
                x = block(x)

        x = self.unpatch(x)

        return x


class ConvAE(nn.Module):
    r"""Creates a convolutional auto-encoder module.

    Arguments:
        encoder: Encoder module.
        decoder: Decoder module.
        saturation: Saturation function applied to latent codes.
        saturation_bound: Bound used by the saturation function.
        noise: Standard deviation of Gaussian noise added during decoding.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        saturation: Optional[str] = "softclip2",
        saturation_bound: float = 5.0,
        noise: float = 0.0,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.saturation = saturation
        self.saturation_bound = saturation_bound
        self.noise = noise

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
        elif self.saturation == "rmsnorm":
            return x * torch.rsqrt(torch.mean(torch.square(x), dim=1, keepdim=True) + 1e-5)
        else:
            raise ValueError(f"unknown saturation '{self.saturation}'")

    def latent_shape(self, resolution: Sequence[int]) -> Tuple[int, ...]:
        r"""Returns the latent tensor shape for a given input resolution.

        Arguments:
            resolution: Spatial dimensions of the input image (L_1, ..., L_N).

        Returns:
            shape: Latent tensor shape (C_z, L_1', ..., L_N'), where each L_i'
                   depends on the encoder's stride and patch size.
        """

        with torch.no_grad():
            dummy = torch.zeros(1, self.encoder.in_channels, *resolution)
            z = self.encoder(dummy)

        return tuple(z.shape[1:])

    def encode(self, x: Tensor) -> Tensor:
        r"""Encodes an image tensor into a latent representation.

        Arguments:
            x: Input image, with shape (B, C_i, L_1, ..., L_N).

        Returns:
            z: Latent code, with shape (B, C_z, L_1', ..., L_N').
        """

        dtype = get_module_dtype(self.encoder)
        z = self.encoder(x.to(dtype))
        z = self.saturate(z)

        return z.to(x.dtype)

    def decode(self, z: Tensor) -> Tensor:
        r"""Decodes a latent code back into an image tensor.

        Arguments:
            z: Latent code, with shape (B, C_z, L_1', ..., L_N').

        Returns:
            x: Reconstructed image, with shape (B, C_o, L_1, ..., L_N).
        """

        dtype = get_module_dtype(self.decoder)

        if self.noise > 0:
            z = z + self.noise * torch.randn_like(z)

        x = self.decoder(z.to(dtype))

        return x.to(z.dtype)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Encodes and reconstructs an image tensor.

        Arguments:
            x: Input image, with shape (B, C_i, L_1, ..., L_N).

        Returns:
            z: Latent code, with shape (B, C_z, L_1', ..., L_N').
            y: Reconstructed image, with shape (B, C_o, L_1, ..., L_N).
        """

        z = self.encode(x)
        y = self.decode(z)

        return z, y


def create_ConvAE(
    in_channels: int,
    out_channels: int,
    lat_channels: int,
    spatial: int = 2,
    noise_level: float = 0.0,
    saturation_bound: float = 5.0,
    saturation: Optional[str] = "softclip2",
    **kwargs,
) -> ConvAE:
    r"""Instantiates a convolutional auto-encoder.

    Arguments:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        lat_channels: Number of latent channels.
        spatial: Number of spatial dimensions.
        noise_level: Standard deviation of Gaussian noise injected at decode time.
        saturation_bound: Bound used by the saturation function.
        saturation: Saturation function applied to latent codes.
        **kwargs: Forwarded to both ConvEncoder and ConvDecoder

    Returns:
        autoencoder: A ConvAE instance.
    """

    encoder = ConvEncoder(
        in_channels=in_channels,
        out_channels=lat_channels,
        spatial=spatial,
        **kwargs,
    )

    decoder = ConvDecoder(
        in_channels=lat_channels,
        out_channels=out_channels,
        spatial=spatial,
        **kwargs,
    )

    return ConvAE(
        encoder,
        decoder,
        saturation=saturation,
        saturation_bound=saturation_bound,
        noise=noise_level,
    )
