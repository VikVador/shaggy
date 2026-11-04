r"""Building blocks for Autoencoders."""

__all__ = [
    "BoundaryConv",
    "FRMSNorm",
    "ModulatedSequential",
    "ResidualBlock",
]

import torch.nn as nn
import torch.nn.functional as F

from azula.nn.layers import ConvNd, RMSNorm
from collections.abc import Sequence
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from typing import Optional, Union


class BoundaryConv(nn.Module):
    r"""Creates a convolution that pads each spatial axis with zeros, periodically or by reflection.

    Arguments:
        in_channels   : Number of input channels C_i.
        out_channels  : Number of output channels C_o.
        spatial       : Number of spatial dimensions N.
        kernel_size   : Kernel size, odd, for every axis or one per axis.
        stride        : Stride, for every axis or one per axis.
        periodic      : Spatial axes padded periodically, e.g. (-1,) for the longitude of (Y, X).
        reflect       : Spatial axes padded by reflection, e.g. (0,) for the latitude of (Y, X).
        identity_init : Whether to initialize the convolution as identity or not.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial: int = 2,
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 1,
        periodic: Sequence[int] = (),
        reflect: Sequence[int] = (),
        identity_init: bool = False,
    ) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * spatial

        # Security
        assert all(-spatial <= axis < spatial for axis in (*periodic, *reflect)), (
            f"ERROR - Axes must be in [{-spatial}, {spatial - 1}], "
            f"got periodic={tuple(periodic)} and reflect={tuple(reflect)}."
        )

        periodic = {axis % spatial for axis in periodic}
        reflect = {axis % spatial for axis in reflect}

        assert not periodic & reflect, (
            f"ERROR - An axis is either periodic or reflected, not both, "
            f"got both for the axes {sorted(periodic & reflect)}."
        )

        # The convolution pads the axes with zeros itself, the other axes are padded beforehand
        self.conv = ConvNd(
            in_channels,
            out_channels,
            spatial=spatial,
            kernel_size=kernel_size,
            stride=stride,
            padding=[0 if a in periodic | reflect else k // 2 for a, k in enumerate(kernel_size)],
            identity_init=identity_init,
        )

        # F.pad lists the paddings from the last axis backwards, and needs one pair for every axis
        self.boundaries = []

        for axis, k in enumerate(kernel_size):
            if axis in periodic | reflect and k > 1:
                padding = [0, 0] * spatial
                padding[2 * (spatial - 1 - axis) : 2 * (spatial - axis)] = [k // 2] * 2
                mode = "circular" if axis in periodic else "reflect"
                self.boundaries.append((padding, mode))

    def forward(self, x: Tensor) -> Tensor:
        r"""Applies the convolution, after padding the periodic and reflected axes.

        Arguments:
            x : Input tensor (B, C_i, L_1, ..., L_N).

        Returns:
            y : Output tensor (B, C_o, L_1 / stride_1, ..., L_N / stride_N).
        """

        for padding, mode in self.boundaries:
            x = F.pad(x, padding, mode=mode)

        return self.conv(x)


class FRMSNorm(RMSNorm):
    r"""Creates a functional root mean square normalization layer.

    References:
        | Skillful joint probabilistic weather forecasting from marginals (Alet et al., 2025)
        | https://arxiv.org/abs/2506.10772

    Arguments:
        channels     : Number of channels C.
        mod_features : Number of modulating features D, or None to disable the modulation.
        spatial      : Number of spatial dimensions N.
        eps          : Numerical stability term.
    """

    modulated = True

    def __init__(
        self,
        channels: int,
        mod_features: Optional[int] = None,
        spatial: int = 2,
        eps: float = 1e-5,
    ) -> None:
        super().__init__(dim=-spatial - 1, eps=eps)

        self.spatial = spatial

        if mod_features:
            self.proj = nn.Linear(mod_features, 2 * channels)
            self.proj.weight.data.mul_(1e-2)
            self.proj.bias.data.zero_()
        else:
            self.proj = None

    def forward(self, x: Tensor, mod: Optional[Tensor] = None) -> Tensor:
        r"""Normalizes the input, then modulates it when a modulation vector is given.

        Arguments:
            x   : Input tensor (B, C, L_1, ..., L_N).
            mod : Modulation vector (B, D).

        Returns:
            y : Output tensor (B, C, L_1, ..., L_N).
        """

        # Normalization
        x = super().forward(x)
        if self.proj is None or mod is None:
            return x

        # Modulation
        gamma, beta = self.proj(mod).chunk(2, dim=-1)
        shape = (*gamma.shape, *(1,) * self.spatial)
        gamma, beta = gamma.reshape(shape), beta.reshape(shape)
        x = (1 + gamma) * x + beta

        return x


class ModulatedSequential(nn.Sequential):
    r"""Creates a sequential container that forwards a modulation vector."""

    modulated = True

    def forward(self, x: Tensor, mod: Optional[Tensor] = None) -> Tensor:
        r"""Applies each module in order, forwarding the modulation to those that accept it.

        Arguments:
            x   : Input tensor (B, C, L_1, ..., L_N).
            mod : Modulation vector (B, D).

        Returns:
            y : Output tensor of the last module.
        """

        for module in self:
            x = module(x, mod) if getattr(module, "modulated", False) else module(x)

        return x


class ResidualBlock(nn.Module):
    r"""Creates a (convolutional) residual block module.

    Arguments:
        channels      : Number of channels C.
        ffn_factor    : Channel expansion factor in the feed-forward network.
        mod_features  : Number of modulating features D, or None to disable the modulation.
        spatial       : Number of spatial dimensions N.
        dropout       : Dropout rate in [0, 1].
        checkpointing : Whether to use gradient checkpointing or not.
        kwargs        : Keyword arguments passed to BoundaryConv (kernel_size, periodic, reflect).
    """

    modulated = True

    def __init__(
        self,
        channels: int,
        ffn_factor: int = 1,
        mod_features: Optional[int] = None,
        spatial: int = 2,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.checkpointing = checkpointing
        self.norm = FRMSNorm(channels, mod_features=mod_features, spatial=spatial)
        self.ffn = nn.Sequential(
            BoundaryConv(
                channels,
                channels * ffn_factor,
                spatial=spatial,
                **kwargs,
            ),
            nn.SiLU(),
            nn.Identity() if dropout is None else nn.Dropout(dropout),
            BoundaryConv(
                channels * ffn_factor,
                channels,
                spatial=spatial,
                **kwargs,
            ),
        )

        # Identity initialization
        self.ffn[-1].conv.weight.data.mul_(1e-2)
        self.ffn[-1].conv.bias.data.zero_()

    def _forward(self, x: Tensor, mod: Optional[Tensor] = None) -> Tensor:
        r"""Adds the output of the feed-forward network to the input, without checkpointing.

        Arguments:
            x   : Input tensor (B, C, L_1, ..., L_N).
            mod : Modulation vector (B, D).

        Returns:
            y : Output tensor (B, C, L_1, ..., L_N).
        """

        return x + self.ffn(self.norm(x, mod))

    def forward(self, x: Tensor, mod: Optional[Tensor] = None) -> Tensor:
        r"""Adds the output of the feed-forward network to the input.

        Arguments:
            x   : Input tensor (B, C, L_1, ..., L_N).
            mod : Modulation vector (B, D).

        Returns:
            y : Output tensor (B, C, L_1, ..., L_N).
        """

        # PyTorch's checkpoint is traced by torch.compile, azula's breaks the graph at every block
        if self.checkpointing and self.training:
            return checkpoint(self._forward, x, mod, use_reentrant=False)
        else:
            return self._forward(x, mod)
