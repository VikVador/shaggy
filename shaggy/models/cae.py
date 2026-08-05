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

from torch import Tensor
from typing import Optional, Sequence, Tuple, Union

from shaggy.layers import (
    ConvNd,
    Patchify,
    ResBlock,
    Unpatchify,
)
from shaggy.models.ae import AutoEncoder


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


class ConvAE(AutoEncoder):
    r"""Creates a convolutional auto-encoder module.

    Arguments:
        encoder: Encoder module.
        decoder: Decoder module.
        saturation: Saturation function applied to latent codes.
        saturation_bound: Bound used by the saturation function.
    """

    def latent_shape(self, resolution: Sequence[int]) -> Tuple[int, ...]:
        r"""Returns the latent tensor shape for a given input resolution.

        Arguments:
            resolution: Spatial dimensions of the input image (L_1, ..., L_N).

        Returns:
            shape: Latent tensor shape (C_z, L_1', ..., L_N'), where each L_i'
                   depends on the encoder's stride and patch size.
        """

        device = next(self.encoder.parameters()).device

        with torch.no_grad():
            dummy = torch.zeros(1, self.encoder.in_channels, *resolution, device=device)
            z = self.encoder(dummy)

        return tuple(z.shape[1:])

    def compression_info(self, input_shape: Sequence[int]) -> Tuple[Tuple[int, ...], int]:
        r"""Returns the bottleneck latent shape and compression factor for a given input shape.

        Arguments:
            input_shape: Full input dimensions (C, L_1, ..., L_N).

        Returns:
            latent: Latent tensor shape (C_z, L_1', ..., L_N').
            factor: Integer compression factor = prod(input_shape) // prod(latent).
        """

        _, *resolution = input_shape
        lat = self.latent_shape(resolution)
        factor = math.prod(input_shape) // math.prod(lat)

        return lat, factor


def create_ConvAE(
    in_channels: int,
    out_channels: int,
    lat_channels: int,
    spatial: int = 2,
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
    )
