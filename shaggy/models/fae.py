r"""Fusion Autoencoder (FAE)."""

__all__ = [
    "FusionEncoder",
    "FusionDecoder",
    "FusionAE",
    "create_FusionAE",
]

import math
import torch
import torch.nn as nn

from azula.nn.utils import get_module_dtype
from torch import Tensor
from typing import Optional, Sequence, Tuple, Union

from shaggy.layers import Compressor, Projector
from shaggy.models.ae import AutoEncoder
from shaggy.models.cae import ConvDecoder, ConvEncoder


class FusionEncoder(nn.Module):
    r"""Creates a Fusion (Convolutional) Encoder that fuses plane (2D) and volume (3D) tensors.

    Arguments:
        in_channels_plane: Number of plane input channels C_p.
        in_channels_volume: Number of volume input channels C_v.
        out_channels: Number of output channels C_o.
        lift_size: Size of the axis the plane tensor is lifted onto.
        lift_blocks: Number of residual blocks in the Projector.
        hid_channels: Numbers of channels at each depth.
        hid_blocks: Numbers of hidden blocks at each depth.
        kernel_size: Kernel size of all convolutions.
        stride: Stride of the downsampling convolutions.
        pixel_shuffle: Whether to use pixel shuffling or not.
        ffn_factor: Channel expansion factor in each FFN.
        patch_size: Patch size applied before the first convolution.
        periodic: Whether the spatial dimensions are periodic or not.
        dropout: Dropout rate in [0, 1].
        checkpointing: Whether to use gradient checkpointing or not.
        identity_init: Initialize down/upsampling convolutions as identity.
    """

    def __init__(
        self,
        in_channels_plane: int,
        in_channels_volume: int,
        out_channels: int,
        lift_size: int,
        lift_blocks: int = 3,
        hid_channels: Sequence[int] = (64, 128, 256),
        hid_blocks: Sequence[int] = (3, 3, 3),
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 2,
        pixel_shuffle: bool = True,
        ffn_factor: int = 1,
        patch_size: Union[int, Sequence[int]] = 1,
        periodic: bool = False,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
        identity_init: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels_plane = in_channels_plane
        self.in_channels_volume = in_channels_volume

        self.projector = Projector(
            channels=in_channels_plane,
            lift_size=lift_size,
            num_blocks=lift_blocks,
            ffn_factor=ffn_factor,
            dropout=dropout,
            checkpointing=checkpointing,
        )

        self.conv = ConvEncoder(
            in_channels=in_channels_plane + in_channels_volume,
            out_channels=out_channels,
            hid_channels=hid_channels,
            hid_blocks=hid_blocks,
            kernel_size=kernel_size,
            stride=stride,
            pixel_shuffle=pixel_shuffle,
            ffn_factor=ffn_factor,
            spatial=3,
            patch_size=patch_size,
            periodic=periodic,
            dropout=dropout,
            checkpointing=checkpointing,
            identity_init=identity_init,
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        r"""
        Arguments:
            x: Plane tensor (B, C_p, X, Y).
            y: Volume tensor (B, C_v, X, Y, Z).

        Dimensions:
            >> x                (B, C_p, X, Y)
            >> projector(x)     (B, C_p, X, Y, Z)
            >> cat([., y])      (B, C_p + C_v, X, Y, Z)
            >> conv             (B, C_o, X', Y', Z')

        Returns:
            z: Latent tensor (B, C_o, X', Y', Z').
        """

        h = torch.cat([self.projector(x), y], dim=1)

        return self.conv(h)


class FusionDecoder(nn.Module):
    r"""Creates a Fusion (Convolutional) Decoder that reconstructs plane (2D) and volume (3D) tensors.

    Arguments:
        in_channels: Number of input channels C_i.
        out_channels_plane: Number of plane output channels C_p.
        out_channels_volume: Number of volume output channels C_v.
        lift_size: Size of the axis the plane tensor is lifted onto.
        lift_blocks: Number of residual blocks in the Projector.
        hid_channels: Numbers of channels at each depth.
        hid_blocks: Numbers of hidden blocks at each depth.
        kernel_size: Kernel size of all convolutions.
        stride: Stride of the upsampling convolutions.
        pixel_shuffle: Whether to use pixel shuffling or not.
        ffn_factor: Channel expansion factor in each FFN.
        patch_size: Patch size applied after the last convolution.
        periodic: Whether the spatial dimensions are periodic or not.
        dropout: Dropout rate in [0, 1].
        checkpointing: Whether to use gradient checkpointing or not.
        identity_init: Initialize down/upsampling convolutions as identity.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels_plane: int,
        out_channels_volume: int,
        lift_size: int,
        lift_blocks: int = 3,
        hid_channels: Sequence[int] = (64, 128, 256),
        hid_blocks: Sequence[int] = (3, 3, 3),
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 2,
        pixel_shuffle: bool = True,
        ffn_factor: int = 1,
        patch_size: Union[int, Sequence[int]] = 1,
        periodic: bool = False,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
        identity_init: bool = True,
    ) -> None:
        super().__init__()

        self.out_channels_plane = out_channels_plane
        self.out_channels_volume = out_channels_volume

        self.compressor = Compressor(
            channels=out_channels_plane,
            lift_size=lift_size,
            num_blocks=lift_blocks,
            ffn_factor=ffn_factor,
            dropout=dropout,
            checkpointing=checkpointing,
        )

        self.conv = ConvDecoder(
            in_channels=in_channels,
            out_channels=out_channels_plane + out_channels_volume,
            hid_channels=hid_channels,
            hid_blocks=hid_blocks,
            kernel_size=kernel_size,
            stride=stride,
            pixel_shuffle=pixel_shuffle,
            ffn_factor=ffn_factor,
            spatial=3,
            patch_size=patch_size,
            periodic=periodic,
            dropout=dropout,
            checkpointing=checkpointing,
            identity_init=identity_init,
        )

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        r"""

        Arguments:
            z: Latent tensor (B, C_i, X', Y', Z').

        Dimensions:
            >> z                     (B, C_i, X', Y', Z')
            >> conv(z)               (B, C_p + C_v, X, Y, Z)
            >> split                 (B, C_p, X, Y, Z), (B, C_v, X, Y, Z)
            >> compressor(plane)     (B, C_p, X, Y)

        Returns:
            x: Plane reconstruction (B, C_p, X, Y).
            y: Volume reconstruction (B, C_v, X, Y, Z).
        """

        h = self.conv(z)

        plane, volume = h.split([self.out_channels_plane, self.out_channels_volume], dim=1)

        return self.compressor(plane), volume


class FusionAE(AutoEncoder):
    r"""Creates a Fusion (Convolutional) Autoencoder (FAE) that fuses plane (2D) and volume (3D) tensors.

    Arguments:
        encoder: FusionEncoder module.
        decoder: FusionDecoder module.
    """

    def encode(self, x: Tensor, y: Tensor) -> Tensor:
        r"""Encodes data in ambient space into a latent representation.

        Arguments:
            x: Plane tensor (B, C_p, X, Y).
            y: Volume tensor (B, C_v, X, Y, Z).

        Returns:
            z: Latent code.
        """

        dtype = get_module_dtype(self.encoder)
        z = self.encoder(x.to(dtype), y.to(dtype))

        return z.to(x.dtype)

    def decode(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Decodes a latent code back into ambient space.

        Arguments:
            z: Latent code.

        Returns:
            x: Reconstructed plane tensor.
            y: Reconstructed volume tensor.
        """

        dtype = get_module_dtype(self.decoder)
        x, y = self.decoder(z.to(dtype))

        return x.to(z.dtype), y.to(z.dtype)

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Encodes and reconstructs data.

        Arguments:
            x: Plane tensor (B, C_p, X, Y).
            y: Volume tensor (B, C_v, X, Y, Z).

        Returns:
            z: Latent code.
            x_hat: Reconstructed plane (2D) tensor.
            y_hat: Reconstructed volume (3D) tensor.
        """

        z = self.encode(x, y)
        x_hat, y_hat = self.decode(z)

        return z, x_hat, y_hat

    def latent(self, resolution: Sequence[int]) -> Tuple[int, ...]:
        r"""Computes the latent shape.

        Arguments:
            resolution: Spatial dimensions of the volume data (X, Y, Z).

        Returns:
            shape: Latent code shape (C_z, X', Y', Z').
        """

        device = next(self.encoder.parameters()).device

        with torch.no_grad():
            dummy_x = torch.zeros(
                1, self.encoder.in_channels_plane, *resolution[:2], device=device
            )
            dummy_y = torch.zeros(1, self.encoder.in_channels_volume, *resolution, device=device)
            z = self.encoder(dummy_x, dummy_y)

        return tuple(z.shape[1:])

    def compression(
        self,
        plane_shape: Sequence[int],
        volume_shape: Sequence[int],
    ) -> Tuple[Tuple[int, ...], int]:
        r"""Computes the compression factor of the autoencoder for a given data shape.

        Arguments:
            plane_shape: Shape of the plane data (C_p, X, Y).
            volume_shape: Shape of the volume data (C_v, X, Y, Z).

        Returns:
            latent: Latent code shape (C_z, X', Y', Z').
            factor: Compression factor.
        """

        _, *resolution = volume_shape
        latent = self.latent(resolution)
        factor = (math.prod(plane_shape) + math.prod(volume_shape)) // math.prod(latent)

        return latent, factor


def create_FusionAE(
    in_channels_plane: int,
    out_channels_plane: int,
    in_channels_volume: int,
    out_channels_volume: int,
    lat_channels: int,
    lift_size: int,
    **kwargs,
) -> FusionAE:
    r"""Instantiates a Fusion (Convolutional) Autoencoder (FAE).

    Arguments:
        in_channels_plane: Number of plane input channels.
        in_channels_volume: Number of volume input channels.
        lat_channels: Number of latent channels.
        lift_size: Size of the axis the plane tensor is lifted onto and folded back from.
        **kwargs: Forwarded to both FusionEncoder and FusionDecoder.

    Returns:
        autoencoder: A FusionAE instance.
    """

    encoder = FusionEncoder(
        in_channels_plane=in_channels_plane,
        in_channels_volume=in_channels_volume,
        out_channels=lat_channels,
        lift_size=lift_size,
        **kwargs,
    )

    decoder = FusionDecoder(
        in_channels=lat_channels,
        out_channels_plane=in_channels_plane,
        out_channels_volume=in_channels_volume,
        lift_size=lift_size,
        **kwargs,
    )

    return FusionAE(encoder, decoder)
