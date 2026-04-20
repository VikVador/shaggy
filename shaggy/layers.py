r"""Shared layers and modules."""

__all__ = [
    "ConvNd",
    "LayerNorm",
    "Patchify",
    "Unpatchify",
]

import torch
import torch.nn as nn

from einops.layers.torch import Rearrange
from torch import Tensor
from typing import Sequence, Union


def ConvNd(
    in_channels: int,
    out_channels: int,
    spatial: int = 2,
    identity_init: bool = False,
    **kwargs,
) -> nn.Module:
    r"""Returns an N-dimensional convolutional layer (1D, 2D, or 3D).

    Arguments:
        in_channels: Number of input channels C_i.
        out_channels: Number of output channels C_o.
        spatial: Number of spatial dimensions N. Must be 1, 2, or 3.
        identity_init: Initialize the convolution weights as a (pseudo-)identity, with small residual noise (scale 1e-2).
        kwargs: Keyword arguments forwarded to the underlying torch.nn.Conv layer.

    Returns:
        conv: Convolutional layer with shape (B, C_i, L_1, ..., L_N) -> (B, C_o, L_1', ..., L_N').
    """

    CONVS = {
        1: nn.Conv1d,
        2: nn.Conv2d,
        3: nn.Conv3d,
    }

    if spatial in CONVS:
        Conv = CONVS[spatial]
    else:
        raise NotImplementedError()

    conv = Conv(in_channels, out_channels, **kwargs)

    if identity_init:
        kernel_size = conv.weight.shape[2:]
        kernel_center = [k // 2 for k in kernel_size]

        eye = torch.zeros_like(conv.weight.data)

        for i in range(out_channels):
            eye[(i, i % in_channels, *kernel_center)] = 1

        conv.weight.data.mul_(1e-2)
        conv.weight.data.add_(eye)

    return conv


class LayerNorm(nn.Module):
    r"""Standardizes features along one or more dimensions.

    Computes: y = (x - mean(x)) / sqrt(var(x) + eps)

    References:
        | Layer Normalization (Lei Ba et al., 2016)
        | https://arxiv.org/abs/1607.06450

    Arguments:
        dim: Dimension(s) to standardize along.
        eps: Numerical stability term added to the variance.
    """

    def __init__(self, dim: Union[int, Sequence[int]], eps: float = 1e-5) -> None:
        super().__init__()

        self.dim = dim if isinstance(dim, int) else tuple(dim)

        self.register_buffer("eps", torch.as_tensor(eps))

    def extra_repr(self) -> str:
        return f"dim={self.dim}"

    def forward(self, x: Tensor) -> Tensor:
        r"""Standardizes the input tensor along the configured dimension(s).

        Arguments:
            x: Input tensor, with shape (*).

        Returns:
            y: Standardized tensor, with shape (*).
        """

        if x.dtype in (torch.float32, torch.float64):
            x32 = x
        else:
            x32 = x.to(dtype=torch.float32)

        variance, mean = torch.var_mean(x32, dim=self.dim, keepdim=True)

        y32 = (x32 - mean) * torch.rsqrt(variance + self.eps)

        return y32.to(dtype=x.dtype)


def Patchify(patch_size: Sequence[int], channel_last: bool = False) -> Rearrange:
    r"""Returns a module that folds spatial patches into the channel dimension.

    Each spatial dimension L_i is split into (L_i / p_i) non-overlapping patches
    of size p_i, and the patch elements are stacked into the channel dimension.
    This reduces each spatial dimension by its patch factor.

    Arguments:
        patch_size: Patch size along each spatial dimension (p_1, ..., p_N).
        channel_last: If True, channels are placed last in the output; otherwise first.

    Returns:
        rearrange: Module mapping (B, C, L_1, ..., L_N) -> (B, C * prod(patch_size), L_1 / p_1, ..., L_N / p_N).
    """

    if len(patch_size) == 1:
        (l,) = patch_size
        if channel_last:
            return Rearrange("... C (L l) -> ... L (C l)", l=l)
        else:
            return Rearrange("... C (L l) -> ... (C l) L", l=l)
    elif len(patch_size) == 2:
        h, w = patch_size
        if channel_last:
            return Rearrange("... C (H h) (W w) -> ... H W (C h w)", h=h, w=w)
        else:
            return Rearrange("... C (H h) (W w) -> ... (C h w) H W", h=h, w=w)
    elif len(patch_size) == 3:
        l, h, w = patch_size
        if channel_last:
            return Rearrange("... C (L l) (H h) (W w) -> ... L H W (C l h w)", l=l, h=h, w=w)
        else:
            return Rearrange("... C (L l) (H h) (W w) -> ... (C l h w) L H W", l=l, h=h, w=w)
    elif len(patch_size) == 4:
        l, h, w, z = patch_size
        if channel_last:
            return Rearrange(
                "... C (L l) (H h) (W w) (Z z) -> ... L H W Z (C l h w z)", l=l, h=h, w=w, z=z
            )
        else:
            return Rearrange(
                "... C (L l) (H h) (W w) (Z z) -> ... (C l h w z) L H W Z", l=l, h=h, w=w, z=z
            )
    else:
        raise NotImplementedError()


def Unpatchify(patch_size: Sequence[int], channel_last: bool = False) -> Rearrange:
    r"""Returns the inverse of Patchify: reconstructs spatial dimensions from channels.

    Each group of prod(patch_size) channels is unfolded back into spatial patch
    elements, expanding each spatial dimension by its patch factor.

    Arguments:
        patch_size: Patch size along each spatial dimension (p_1, ..., p_N).
        channel_last: If True, channels are expected last in the input; otherwise first.

    Returns:
        rearrange: Module mapping (B, C * prod(patch_size), L_1 / p_1, ..., L_N / p_N) -> (B, C, L_1, ..., L_N).
    """

    if len(patch_size) == 1:
        (l,) = patch_size
        if channel_last:
            return Rearrange("... L (C l) -> ... C (L l)", l=l)
        else:
            return Rearrange("... (C l) L -> ... C (L l)", l=l)
    elif len(patch_size) == 2:
        h, w = patch_size
        if channel_last:
            return Rearrange("... H W (C h w) -> ... C (H h) (W w)", h=h, w=w)
        else:
            return Rearrange("... (C h w) H W -> ... C (H h) (W w)", h=h, w=w)
    elif len(patch_size) == 3:
        l, h, w = patch_size
        if channel_last:
            return Rearrange("... L H W (C l h w) -> ... C (L l) (H h) (W w)", l=l, h=h, w=w)
        else:
            return Rearrange("... (C l h w) L H W -> ... C (L l) (H h) (W w)", l=l, h=h, w=w)
    elif len(patch_size) == 4:
        l, h, w, z = patch_size
        if channel_last:
            return Rearrange(
                "... L H W Z (C l h w z) -> ... C (L l) (H h) (W w) (Z z)", l=l, h=h, w=w, z=z
            )
        else:
            return Rearrange(
                "... (C l h w z) L H W Z -> ... C (L l) (H h) (W w) (Z z)", l=l, h=h, w=w, z=z
            )
    else:
        raise NotImplementedError()
