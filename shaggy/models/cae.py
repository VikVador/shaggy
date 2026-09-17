r"""Convolutional Autoencoder (CAE)."""

__all__ = [
    "ConvEncoder",
    "ConvDecoder",
    "ConvAE",
    "create_ConvAE",
]

import math
import torch
import torch.nn as nn

from azula.nn.layers import ConvNd, Patchify, Unpatchify
from collections.abc import Sequence
from torch import Tensor
from typing import Any, Optional, Union

from shaggy.layers import FRMSNorm, ModulatedSequential, ResidualTrunk
from shaggy.models.ae import AutoEncoder


def broadcast_to_axes(value: Union[int, Sequence[int]], spatial: int) -> tuple[int, ...]:
    r"""Broadcasts an integer, or a sequence of integers, to one value per spatial axis."""
    if isinstance(value, int):
        return (value,) * spatial

    # Security
    assert len(value) == spatial, (
        f"ERROR (broadcast_to_axes) | Expected {spatial} values, one per axis, got {len(value)}."
    )

    return tuple(value)


class ConvEncoder(nn.Module):
    r"""Creates a convolutional encoder module.

    Arguments:
        in_channels: Number of input channels C_i.
        out_channels: Number of output channels C_o.
        hid_channels: Number of channels at each depth.
        hid_blocks: Number of residual blocks at each depth.
        hid_groups: Number of residual groups at each depth, or 0 for a flat stack.
        kernel_size: Kernel size of all convolutions.
        stride: Stride of the downsampling convolutions.
        pixel_shuffle: Whether to downsample with pixel shuffling or not.
        ffn_factor: Channel expansion factor in the feed-forward networks.
        mod_features: Number of modulating features D, or None to disable the modulation.
        spatial: Number of spatial dimensions N.
        patch_size: Patch size applied before the first convolution.
        periodic: Whether the spatial dimensions are periodic or not.
        dropout: Dropout rate in [0, 1].
        checkpointing: Whether to use gradient checkpointing or not.
        identity_init: Whether to initialize projection and resampling layers as identity or not.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hid_channels: Sequence[int] = (64, 128, 256),
        hid_blocks: Sequence[int] = (3, 3, 3),
        hid_groups: Optional[Sequence[int]] = None,
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 2,
        pixel_shuffle: bool = True,
        ffn_factor: int = 1,
        mod_features: Optional[int] = None,
        spatial: int = 2,
        patch_size: Union[int, Sequence[int]] = 1,
        periodic: bool = False,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
        identity_init: bool = True,
    ) -> None:
        super().__init__()

        if hid_groups is None:
            hid_groups = [0] * len(hid_channels)

        kernel_size = broadcast_to_axes(kernel_size, spatial)
        stride = broadcast_to_axes(stride, spatial)
        patch_size = broadcast_to_axes(patch_size, spatial)

        # Security
        assert len(hid_channels) == len(hid_blocks) == len(hid_groups), (
            "ERROR (ConvEncoder) | hid_channels, hid_blocks and hid_groups must match in length."
        )

        assert all(k % 2 == 1 for k in kernel_size), (
            "ERROR (ConvEncoder) | Kernel sizes must be odd to preserve the spatial dimensions."
        )

        kwargs = dict(
            kernel_size=kernel_size,
            padding=tuple(k // 2 for k in kernel_size),
            padding_mode="circular" if periodic else "zeros",
        )

        self.in_channels = in_channels
        self.scale = tuple(p * s ** (len(hid_channels) - 1) for p, s in zip(patch_size, stride))

        self.patch = Patchify(patch_shape=patch_size)

        self.in_proj = ConvNd(
            math.prod(patch_size) * in_channels,
            hid_channels[0],
            spatial=spatial,
            identity_init=identity_init,
            **kwargs,
        )

        self.descent = ModulatedSequential()

        for i, (num_blocks, num_groups) in enumerate(zip(hid_blocks, hid_groups)):
            if i > 0:
                self.descent.append(
                    FRMSNorm(hid_channels[i - 1], mod_features=mod_features, spatial=spatial)
                )

                if pixel_shuffle:
                    self.descent.append(
                        nn.Sequential(
                            Patchify(patch_shape=stride),
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
                    self.descent.append(
                        ConvNd(
                            hid_channels[i - 1],
                            hid_channels[i],
                            spatial=spatial,
                            stride=stride,
                            identity_init=identity_init,
                            **kwargs,
                        )
                    )

            self.descent.append(
                ResidualTrunk(
                    hid_channels[i],
                    num_blocks=num_blocks,
                    num_groups=num_groups,
                    spatial=spatial,
                    ffn_factor=ffn_factor,
                    mod_features=mod_features,
                    dropout=dropout,
                    checkpointing=checkpointing,
                    **kwargs,
                )
            )

        self.out_norm = FRMSNorm(hid_channels[-1], mod_features=mod_features, spatial=spatial)

        self.out_proj = ConvNd(
            hid_channels[-1],
            out_channels,
            spatial=spatial,
            identity_init=identity_init,
            **kwargs,
        )

    def forward(self, x: Tensor, mod: Optional[Tensor] = None) -> Tensor:
        r"""
        Arguments:
            x: Input tensor (B, C_i, L_1, ..., L_N).
            mod: Modulation vector (B, D).

        Returns:
            Output tensor (B, C_o, L_1 / scale_1, ..., L_N / scale_N).
        """
        x = self.descent(self.in_proj(self.patch(x)), mod)
        x = self.out_proj(self.out_norm(x, mod))
        return x


class ConvDecoder(nn.Module):
    r"""Creates a convolutional decoder module.

    Arguments:
        in_channels: Number of input channels C_i.
        out_channels: Number of output channels C_o.
        hid_channels: Number of channels at each depth.
        hid_blocks: Number of residual blocks at each depth.
        hid_groups: Number of residual groups at each depth, or 0 for a flat stack.
        kernel_size: Kernel size of all convolutions.
        stride: Stride of the upsampling convolutions.
        pixel_shuffle: Whether to upsample with pixel shuffling or not.
        ffn_factor: Channel expansion factor in the feed-forward networks.
        mod_features: Number of modulating features D, or None to disable the modulation.
        spatial: Number of spatial dimensions N.
        patch_size: Patch size applied after the last convolution.
        periodic: Whether the spatial dimensions are periodic or not.
        dropout: Dropout rate in [0, 1].
        checkpointing: Whether to use gradient checkpointing or not.
        identity_init: Whether to initialize projection and resampling layers as identity or not.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hid_channels: Sequence[int] = (64, 128, 256),
        hid_blocks: Sequence[int] = (3, 3, 3),
        hid_groups: Optional[Sequence[int]] = None,
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 2,
        pixel_shuffle: bool = True,
        ffn_factor: int = 1,
        mod_features: Optional[int] = None,
        spatial: int = 2,
        patch_size: Union[int, Sequence[int]] = 1,
        periodic: bool = False,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
        identity_init: bool = True,
    ) -> None:
        super().__init__()

        if hid_groups is None:
            hid_groups = [0] * len(hid_channels)

        kernel_size = broadcast_to_axes(kernel_size, spatial)
        stride = broadcast_to_axes(stride, spatial)
        patch_size = broadcast_to_axes(patch_size, spatial)

        # Security
        assert len(hid_channels) == len(hid_blocks) == len(hid_groups), (
            "ERROR (ConvDecoder) | hid_channels, hid_blocks and hid_groups must match in length."
        )

        assert all(k % 2 == 1 for k in kernel_size), (
            "ERROR (ConvDecoder) | Kernel sizes must be odd to preserve the spatial dimensions."
        )

        kwargs = dict(
            kernel_size=kernel_size,
            padding=tuple(k // 2 for k in kernel_size),
            padding_mode="circular" if periodic else "zeros",
        )

        self.scale = tuple(p * s ** (len(hid_channels) - 1) for p, s in zip(patch_size, stride))

        self.in_proj = ConvNd(
            in_channels,
            hid_channels[-1],
            spatial=spatial,
            identity_init=identity_init,
            **kwargs,
        )

        self.ascent = ModulatedSequential()

        for i in reversed(range(len(hid_channels))):
            self.ascent.append(
                ResidualTrunk(
                    hid_channels[i],
                    num_blocks=hid_blocks[i],
                    num_groups=hid_groups[i],
                    spatial=spatial,
                    ffn_factor=ffn_factor,
                    mod_features=mod_features,
                    dropout=dropout,
                    checkpointing=checkpointing,
                    **kwargs,
                )
            )

            if i > 0:
                self.ascent.append(
                    FRMSNorm(hid_channels[i], mod_features=mod_features, spatial=spatial)
                )

                if pixel_shuffle:
                    self.ascent.append(
                        nn.Sequential(
                            ConvNd(
                                hid_channels[i],
                                hid_channels[i - 1] * math.prod(stride),
                                spatial=spatial,
                                identity_init=identity_init,
                                **kwargs,
                            ),
                            Unpatchify(patch_shape=stride),
                        )
                    )
                else:
                    self.ascent.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=stride, mode="nearest"),
                            ConvNd(
                                hid_channels[i],
                                hid_channels[i - 1],
                                spatial=spatial,
                                identity_init=identity_init,
                                **kwargs,
                            ),
                        )
                    )

        self.out_norm = FRMSNorm(hid_channels[0], mod_features=mod_features, spatial=spatial)

        self.out_proj = ConvNd(
            hid_channels[0],
            math.prod(patch_size) * out_channels,
            spatial=spatial,
            identity_init=identity_init,
            **kwargs,
        )

        self.unpatch = Unpatchify(patch_shape=patch_size)

    def forward(self, x: Tensor, mod: Optional[Tensor] = None) -> Tensor:
        r"""
        Arguments:
            x: Input tensor (B, C_i, L_1, ..., L_N).
            mod: Modulation vector (B, D).

        Returns:
            Output tensor (B, C_o, L_1 * scale_1, ..., L_N * scale_N).
        """
        x = self.ascent(self.in_proj(x), mod)
        x = self.unpatch(self.out_proj(self.out_norm(x, mod)))
        return x


class ConvAE(AutoEncoder):
    r"""Creates a Convolutional Autoencoder (CAE).

    Arguments:
        encoder: Encoder module.
        decoder: Decoder module.
    """

    def latent(self, resolution: Sequence[int]) -> tuple[int, ...]:
        r"""Computes the latent shape.

        Arguments:
            resolution: Spatial dimensions of the data (L_1, ..., L_N).

        Returns:
            shape: Latent code shape (C_z, L_1', ..., L_N').
        """

        device = next(self.encoder.parameters()).device

        with torch.no_grad():
            dummy = torch.zeros(1, self.encoder.in_channels, *resolution, device=device)
            z = self.encoder(dummy)

        return tuple(z.shape[1:])

    def compression(self, input_shape: Sequence[int]) -> tuple[tuple[int, ...], int]:
        r"""Computes the compression factor of the autoencoder for a given data shape.

        Arguments:
            input_shape: Shape of the data (C, L_1, ..., L_N).

        Returns:
            latent: Latent code shape (C_z, L_1', ..., L_N').
            factor: Compression factor.
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
        in_channels: Number of input channels C_i.
        out_channels: Number of output channels C_o.
        lat_channels: Number of latent channels C_z.
        spatial: Number of spatial dimensions N.
        config_encoder: Keyword arguments overriding kwargs for the encoder.
        config_decoder: Keyword arguments overriding kwargs for the decoder.
        kwargs: Keyword arguments shared by ConvEncoder and ConvDecoder.

    Returns:
        autoencoder: A ConvAE instance.
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
        f"ERROR (create_ConvAE) | Encoder downsamples by {encoder.scale} "
        f"but decoder upsamples by {decoder.scale}."
    )

    return ConvAE(encoder, decoder)
