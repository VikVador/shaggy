r"""Building blocks for Autoencoders."""

__all__ = [
    "ResBlock",
    "Projector",
    "Compressor",
]

import torch.nn as nn

from azula.nn.layers import ConvNd, LayerNorm
from azula.nn.utils import checkpoint
from torch import Tensor
from typing import Optional


class ResBlock(nn.Module):
    r"""Creates a convolutional residual block module.

    Arguments:
        channels: Number of channels C.
        ffn_factor: Channel expansion factor in the feed-forward network.
        spatial: Number of spatial dimensions N.
        dropout: Dropout rate in [0, 1].
        checkpointing: Whether to use gradient checkpointing or not.
        kwargs: Keyword arguments passed to azula.nn.layers.ConvNd.
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

        self.norm = LayerNorm(dim=-spatial - 1)

        self.ffn = nn.Sequential(
            ConvNd(channels, ffn_factor * channels, spatial=spatial, **kwargs),
            nn.SiLU(),
            nn.Identity() if dropout is None else nn.Dropout(dropout),
            ConvNd(ffn_factor * channels, channels, spatial=spatial, **kwargs),
        )

        self.ffn[-1].weight.data.mul_(1e-2)

    def _forward(self, x: Tensor) -> Tensor:
        y = self.norm(x)
        y = self.ffn(y)
        return x + y

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: Input tensor (B, C, L_1, ..., L_N).

        Returns:
            Output tensor (B, C, L_1, ..., L_N).
        """

        if self.checkpointing:
            return checkpoint(self._forward, reentrant=not self.training)(x)
        else:
            return self._forward(x)


class Projector(nn.Module):
    r"""Lifts a plane tensor onto a new trailing axis.

    Lifting:
        >> x        (B, C, X, Y)
        >> lift(x)  (B, C * L, X, Y)
        >> view     (B, C, L, X, Y)
        >> permute  (B, C, X, Y, L)

    Arguments:
        channels: Number of channels C.
        lift_size: Size of the axis the tensor is lifted onto.
        num_blocks: Number of residual blocks applied in the lifted space.
        ffn_factor: Channel expansion factor in each FFN.
        dropout: Dropout rate in [0, 1].
        checkpointing: Whether to use gradient checkpointing or not.
    """

    def __init__(
        self,
        channels: int,
        lift_size: int,
        num_blocks: int = 3,
        ffn_factor: int = 1,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
    ) -> None:
        super().__init__()

        self.channels = channels
        self.lift_size = lift_size

        self.lift = ConvNd(channels, lift_size * channels, spatial=2, kernel_size=1)

        self.blocks = nn.Sequential(*[
            ResBlock(
                lift_size * channels,
                ffn_factor=ffn_factor,
                spatial=2,
                dropout=dropout,
                checkpointing=checkpointing,
                kernel_size=1,
            )
            for _ in range(num_blocks)
        ])

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: Plane tensor (B, C, X, Y).

        Returns:
            Volume tensor (B, C, X, Y, L).
        """

        b, _, nx, ny = x.shape

        x = self.lift(x)
        x = self.blocks(x)
        x = x.view(b, self.channels, self.lift_size, nx, ny)
        x = x.permute(0, 1, 3, 4, 2).contiguous()

        return x


class Compressor(nn.Module):
    r"""Folds the trailing axis of a volume tensor into channels.

    Compressing:
        >> x        (B, C, X, Y, L)
        >> permute  (B, C, L, X, Y)
        >> reshape  (B, C * L, X, Y)
        >> project  (B, C, X, Y)


    Arguments:
        channels: Number of channels C.
        lift_size: Size of the axis folded into the channel dimension.
        num_blocks: Number of residual blocks applied in the folded (C * lift_size) space.
        ffn_factor: Channel expansion factor in each ResBlock's FFN.
        dropout: Dropout rate in [0, 1].
        checkpointing: Whether to use gradient checkpointing or not.
    """

    def __init__(
        self,
        channels: int,
        lift_size: int,
        num_blocks: int = 3,
        ffn_factor: int = 1,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(*[
            ResBlock(
                lift_size * channels,
                ffn_factor=ffn_factor,
                spatial=2,
                dropout=dropout,
                checkpointing=checkpointing,
                kernel_size=1,
            )
            for _ in range(num_blocks)
        ])

        self.project = ConvNd(lift_size * channels, channels, spatial=2, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: Volume tensor (B, C, X, Y, L).

        Returns:
            Plane tensor (B, C, X, Y).
        """

        b, c, nx, ny, nz = x.shape

        x = x.permute(0, 1, 4, 2, 3).reshape(b, c * nz, nx, ny)
        x = self.blocks(x)
        x = self.project(x)

        return x
