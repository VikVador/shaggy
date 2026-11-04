r"""Convolutional Autoencoder (CAE)."""

__all__ = [
    "ConvEncoder",
    "ConvDecoder",
    "ConvAE",
    "create_ConvAE",
]

import math
import torch.nn as nn

from azula.nn.layers import ConvNd, Patchify, Unpatchify
from collections.abc import Sequence
from torch import Tensor
from typing import Any, Optional, Union

from shaggy.layers import BoundaryConv, FRMSNorm, ModulatedSequential, ResidualBlock
from shaggy.models.ae import AutoEncoder


def broadcast_to_axes(value: Union[int, Sequence[int]], spatial: int) -> tuple[int, ...]:
    r"""Broadcasts an integer, or a sequence of integers, to one value per spatial axis.

    Arguments:
        value   : An int shared by every axis, or one int per axis.
        spatial : Number of spatial dimensions N.

    Returns:
        values : One value per axis.
    """

    if isinstance(value, int):
        return (value,) * spatial

    # Security
    assert len(value) == spatial, (
        f"ERROR - Expected {spatial} values, one per axis, got {len(value)}."
    )

    return tuple(value)


def broadcast_to_levels(
    value: Union[int, Sequence[Union[int, Sequence[int]]]],
    levels: int,
    spatial: int,
) -> list[tuple[int, ...]]:
    r"""Broadcasts a stride to one value per level and per axis.

    Arguments:
        value   : An int or one int per axis, shared by every level, or one such stride per level.
        levels  : Number of resampling levels.
        spatial : Number of spatial dimensions N.

    Returns:
        strides : One stride per level, each with one value per axis.
    """

    if isinstance(value, int) or all(isinstance(v, int) for v in value):
        return [broadcast_to_axes(value, spatial)] * levels

    # Security
    assert len(value) == levels, (
        f"ERROR - Expected {levels} strides, one per level, got {len(value)}."
    )

    return [broadcast_to_axes(v, spatial) for v in value]


class ConvEncoder(nn.Module):
    r"""Creates a convolutional encoder module.

    Arguments:
        in_channels   : Number of input channels C_i.
        out_channels  : Number of output channels C_o.
        mix_channels  : Number of channels of a pointwise convolution mixing the inputs first, or None.
        hid_channels  : Number of channels at each depth.
        hid_blocks    : Number of residual blocks at each depth.
        kernel_size   : Kernel size of all convolutions.
        stride        : Stride of the downsampling convolutions, for every level or one per level.
        pixel_shuffle : Whether to downsample with pixel shuffling or not.
        ffn_factor    : Channel expansion factor in the feed-forward networks.
        mod_features  : Number of modulating features D, or None to disable the modulation.
        spatial       : Number of spatial dimensions N.
        patch_size    : Patch size applied before the first convolution.
        periodic      : Spatial axes padded periodically, counted from 0 or from the end (-1).
        reflect       : Spatial axes padded by reflection, counted from 0 or from the end (-1).
        dropout       : Dropout rate in [0, 1].
        checkpointing : Whether to use gradient checkpointing or not.
        identity_init : Whether to initialize projection and resampling layers as identity or not.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mix_channels: Optional[int] = None,
        hid_channels: Sequence[int] = (64, 128, 256),
        hid_blocks: Sequence[int] = (3, 3, 3),
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[Union[int, Sequence[int]]]] = 2,
        pixel_shuffle: bool = True,
        ffn_factor: int = 1,
        mod_features: Optional[int] = None,
        spatial: int = 2,
        patch_size: Union[int, Sequence[int]] = 1,
        periodic: Sequence[int] = (),
        reflect: Sequence[int] = (),
        dropout: Optional[float] = None,
        checkpointing: bool = False,
        identity_init: bool = True,
    ) -> None:
        super().__init__()

        kernel_size = broadcast_to_axes(kernel_size, spatial)
        strides = broadcast_to_levels(stride, len(hid_channels) - 1, spatial)
        patch_size = broadcast_to_axes(patch_size, spatial)

        # Security
        assert len(hid_channels) == len(hid_blocks), (
            "ERROR - hid_channels and hid_blocks must match in length."
        )

        assert all(k % 2 == 1 for k in kernel_size), (
            "ERROR - Kernel sizes must be odd to preserve the spatial dimensions."
        )

        kwargs = dict(kernel_size=kernel_size, periodic=periodic, reflect=reflect)

        self.scale = tuple(p * math.prod(s[a] for s in strides) for a, p in enumerate(patch_size))

        # The pointwise convolution mixes the input channels at full resolution, such that fewer
        # channels reach the first convolution
        if mix_channels is None:
            self.mix = nn.Identity()
        else:
            self.mix = ConvNd(
                in_channels,
                mix_channels,
                spatial=spatial,
                kernel_size=1,
                identity_init=identity_init,
            )
            in_channels = mix_channels

        self.patch = Patchify(patch_shape=patch_size)

        self.in_proj = BoundaryConv(
            math.prod(patch_size) * in_channels,
            hid_channels[0],
            spatial=spatial,
            identity_init=identity_init,
            **kwargs,
        )

        self.descent = ModulatedSequential()

        for i, num_blocks in enumerate(hid_blocks):
            if i > 0:
                self.descent.append(
                    FRMSNorm(hid_channels[i - 1], mod_features=mod_features, spatial=spatial)
                )

                if pixel_shuffle:
                    self.descent.append(
                        nn.Sequential(
                            Patchify(patch_shape=strides[i - 1]),
                            BoundaryConv(
                                hid_channels[i - 1] * math.prod(strides[i - 1]),
                                hid_channels[i],
                                spatial=spatial,
                                identity_init=identity_init,
                                **kwargs,
                            ),
                        )
                    )
                else:
                    self.descent.append(
                        BoundaryConv(
                            hid_channels[i - 1],
                            hid_channels[i],
                            spatial=spatial,
                            stride=strides[i - 1],
                            identity_init=identity_init,
                            **kwargs,
                        )
                    )

            self.descent.append(
                ModulatedSequential(*[
                    ResidualBlock(
                        hid_channels[i],
                        ffn_factor=ffn_factor,
                        mod_features=mod_features,
                        spatial=spatial,
                        dropout=dropout,
                        checkpointing=checkpointing,
                        **kwargs,
                    )
                    for _ in range(num_blocks)
                ])
            )

        self.out_norm = FRMSNorm(hid_channels[-1], mod_features=mod_features, spatial=spatial)

        self.out_proj = BoundaryConv(
            hid_channels[-1],
            out_channels,
            spatial=spatial,
            identity_init=identity_init,
            **kwargs,
        )

    def forward(self, x: Tensor, mod: Optional[Tensor] = None) -> Tensor:
        r"""Encodes the input into a latent code.

        Arguments:
            x   : Input tensor (B, C_i, L_1, ..., L_N).
            mod : Modulation vector (B, D).

        Returns:
            y : Output tensor (B, C_o, L_1 / scale_1, ..., L_N / scale_N).
        """

        x = self.descent(self.in_proj(self.patch(self.mix(x))), mod)
        x = self.out_proj(self.out_norm(x, mod))

        return x


class ConvDecoder(nn.Module):
    r"""Creates a convolutional decoder module.

    Arguments:
        in_channels   : Number of input channels C_i.
        out_channels  : Number of output channels C_o.
        hid_channels  : Number of channels at each depth.
        hid_blocks    : Number of residual blocks at each depth.
        kernel_size   : Kernel size of all convolutions.
        stride        : Stride of the upsampling convolutions, for every level or one per level.
        pixel_shuffle : Whether to upsample with pixel shuffling or not.
        ffn_factor    : Channel expansion factor in the feed-forward networks.
        mod_features  : Number of modulating features D, or None to disable the modulation.
        spatial       : Number of spatial dimensions N.
        patch_size    : Patch size applied after the last convolution.
        periodic      : Spatial axes padded periodically, counted from 0 or from the end (-1).
        reflect       : Spatial axes padded by reflection, counted from 0 or from the end (-1).
        dropout       : Dropout rate in [0, 1].
        checkpointing : Whether to use gradient checkpointing or not.
        identity_init : Whether to initialize projection and resampling layers as identity or not.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hid_channels: Sequence[int] = (64, 128, 256),
        hid_blocks: Sequence[int] = (3, 3, 3),
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[Union[int, Sequence[int]]]] = 2,
        pixel_shuffle: bool = True,
        ffn_factor: int = 1,
        mod_features: Optional[int] = None,
        spatial: int = 2,
        patch_size: Union[int, Sequence[int]] = 1,
        periodic: Sequence[int] = (),
        reflect: Sequence[int] = (),
        dropout: Optional[float] = None,
        checkpointing: bool = False,
        identity_init: bool = True,
    ) -> None:
        super().__init__()

        kernel_size = broadcast_to_axes(kernel_size, spatial)
        strides = broadcast_to_levels(stride, len(hid_channels) - 1, spatial)
        patch_size = broadcast_to_axes(patch_size, spatial)

        # Security
        assert len(hid_channels) == len(hid_blocks), (
            "ERROR - hid_channels and hid_blocks must match in length."
        )

        assert all(k % 2 == 1 for k in kernel_size), (
            "ERROR - Kernel sizes must be odd to preserve the spatial dimensions."
        )

        kwargs = dict(kernel_size=kernel_size, periodic=periodic, reflect=reflect)

        self.scale = tuple(p * math.prod(s[a] for s in strides) for a, p in enumerate(patch_size))

        self.in_proj = BoundaryConv(
            in_channels,
            hid_channels[-1],
            spatial=spatial,
            identity_init=identity_init,
            **kwargs,
        )

        self.ascent = ModulatedSequential()

        for i in reversed(range(len(hid_channels))):
            self.ascent.append(
                ModulatedSequential(*[
                    ResidualBlock(
                        hid_channels[i],
                        ffn_factor=ffn_factor,
                        mod_features=mod_features,
                        spatial=spatial,
                        dropout=dropout,
                        checkpointing=checkpointing,
                        **kwargs,
                    )
                    for _ in range(hid_blocks[i])
                ])
            )

            if i > 0:
                self.ascent.append(
                    FRMSNorm(hid_channels[i], mod_features=mod_features, spatial=spatial)
                )

                if pixel_shuffle:
                    self.ascent.append(
                        nn.Sequential(
                            BoundaryConv(
                                hid_channels[i],
                                hid_channels[i - 1] * math.prod(strides[i - 1]),
                                spatial=spatial,
                                identity_init=identity_init,
                                **kwargs,
                            ),
                            Unpatchify(patch_shape=strides[i - 1]),
                        )
                    )
                else:
                    self.ascent.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=strides[i - 1], mode="nearest"),
                            BoundaryConv(
                                hid_channels[i],
                                hid_channels[i - 1],
                                spatial=spatial,
                                identity_init=identity_init,
                                **kwargs,
                            ),
                        )
                    )

        self.out_norm = FRMSNorm(hid_channels[0], mod_features=mod_features, spatial=spatial)

        self.out_proj = BoundaryConv(
            hid_channels[0],
            math.prod(patch_size) * out_channels,
            spatial=spatial,
            identity_init=identity_init,
            **kwargs,
        )

        self.unpatch = Unpatchify(patch_shape=patch_size)

    def forward(self, x: Tensor, mod: Optional[Tensor] = None) -> Tensor:
        r"""Decodes a latent code into the ambient space.

        Arguments:
            x   : Input tensor (B, C_i, L_1, ..., L_N).
            mod : Modulation vector (B, D).

        Returns:
            y : Output tensor (B, C_o, L_1 * scale_1, ..., L_N * scale_N).
        """

        x = self.ascent(self.in_proj(x), mod)
        x = self.unpatch(self.out_proj(self.out_norm(x, mod)))

        return x


class ConvAE(AutoEncoder):
    r"""Creates a Convolutional Autoencoder (CAE).

    Arguments:
        encoder : Encoder module.
        decoder : Decoder module.
    """

    def latent(self, resolution: Sequence[int]) -> tuple[int, ...]:
        r"""Computes the latent shape.

        Arguments:
            resolution : Spatial dimensions of the data (L_1, ..., L_N).

        Returns:
            shape : Latent code shape (C_z, L_1', ..., L_N').
        """

        # The encoder divides each spatial dimension by its own scale
        return (
            self.encoder.out_proj.conv.out_channels,
            *(length // scale for length, scale in zip(resolution, self.encoder.scale)),
        )

    def compression(self, input_shape: Sequence[int]) -> tuple[tuple[int, ...], int]:
        r"""Computes the compression factor of the autoencoder for a given data shape.

        Arguments:
            input_shape : Shape of the data (C, L_1, ..., L_N).

        Returns:
            latent : Latent code shape (C_z, L_1', ..., L_N').
            factor : Compression factor.
        """

        _, *resolution = input_shape
        latent = self.latent(resolution)
        factor = math.prod(input_shape) // math.prod(latent)

        return latent, factor


def create_ConvAE(
    in_channels: int,
    out_channels: int,
    lat_channels: int,
    spatial: int = 2,
    config_encoder: Optional[dict[str, Any]] = None,
    config_decoder: Optional[dict[str, Any]] = None,
    **kwargs,
) -> ConvAE:
    r"""Instantiates a Convolutional Autoencoder (CAE).

    Arguments:
        in_channels    : Number of input channels C_i.
        out_channels   : Number of output channels C_o.
        lat_channels   : Number of latent channels C_z.
        spatial        : Number of spatial dimensions N.
        config_encoder : Keyword arguments overriding kwargs for the encoder.
        config_decoder : Keyword arguments overriding kwargs for the decoder.
        kwargs         : Keyword arguments shared by ConvEncoder and ConvDecoder.

    Returns:
        autoencoder : A ConvAE instance.
    """

    encoder = ConvEncoder(
        in_channels=in_channels,
        out_channels=lat_channels,
        spatial=spatial,
        **{**kwargs, **(config_encoder or {})},
    )

    decoder = ConvDecoder(
        in_channels=lat_channels,
        out_channels=out_channels,
        spatial=spatial,
        **{**kwargs, **(config_decoder or {})},
    )

    # Security
    assert encoder.scale == decoder.scale, (
        f"ERROR - Encoder downsamples by {encoder.scale} but decoder upsamples by {decoder.scale}."
    )

    return ConvAE(encoder, decoder)
