r"""Shared layers and modules."""

__all__ = [
    "ConvNd",
    "LayerNorm",
    "Patchify",
    "Unpatchify",
    "Residual",
    "ResBlock",
    "GraphResBlock",
    "GraphPool",
]

import torch
import torch.nn as nn

from einops.layers.torch import Rearrange
from torch import Tensor
from typing import Optional, Sequence, Union

from shaggy.utils import checkpoint


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


def _in_degree(edge_index: Tensor, num_nodes: int) -> Tensor:
    r"""Counts incoming edges per node, for mean (rather than sum) aggregation.

    Sum aggregation over a K-means cluster hierarchy would let a target node's output
    magnitude be dominated by its (input-independent) cluster size rather than by the
    actual source features, which stalls training. Dividing by in-degree fixes this.

    Arguments:
        edge_index: Edge index, with shape (2, E).
        num_nodes: Number of target nodes N.

    Returns:
        degree: In-degree per node, clamped to >= 1, with shape (1, N, 1).
    """
    degree = torch.zeros(num_nodes)
    degree.index_add_(0, edge_index[1], torch.ones(edge_index.shape[1]))
    return degree.clamp(min=1).view(1, num_nodes, 1)


class GraphResBlock(nn.Module):
    r"""Creates a residual message-passing block operating on a single mesh level.

    Analogous to :class:`ResBlock`, but the spatial convolution is replaced by an
    edge-conditioned message passing step (an Interaction Network, following
    Njord/GraphCast-style GNNs) over a fixed graph.

    Arguments:
        channels: Number of node features C.
        edge_index: Intra-level edge index, with shape (2, E).
        num_nodes: Number of nodes N at this level.
        ffn_factor: Channel expansion factor in the MLPs.
        dropout: Dropout rate in [0, 1].
        checkpointing: Whether to use gradient checkpointing or not.
    """

    def __init__(
        self,
        channels: int,
        edge_index: Tensor,
        num_nodes: int,
        ffn_factor: int = 1,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
    ) -> None:
        super().__init__()

        self.checkpointing = checkpointing

        self.register_buffer("edge_index", edge_index)
        self.register_buffer("degree", _in_degree(edge_index, num_nodes))

        self.norm = LayerNorm(dim=-1)

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * channels, ffn_factor * channels),
            nn.SiLU(),
            nn.Linear(ffn_factor * channels, channels),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(2 * channels, ffn_factor * channels),
            nn.SiLU(),
            nn.Identity() if dropout is None else nn.Dropout(dropout),
            nn.Linear(ffn_factor * channels, channels),
        )

        self.node_mlp[-1].weight.data.mul_(1e-2)

    def _forward(self, h: Tensor) -> Tensor:
        r"""Applies layer norm, edge update, node aggregation/update, and residual addition.

        Arguments:
            h: Node features, with shape (B, N, C).

        Returns:
            Output node features, with shape (B, N, C).
        """

        y = self.norm(h)

        src, dst = self.edge_index

        edges = self.edge_mlp(torch.cat([y[:, src], y[:, dst]], dim=-1))

        agg = y.new_zeros(y.shape[0], y.shape[1], edges.shape[-1])
        agg = agg.index_add_(1, dst, edges) / self.degree

        out = self.node_mlp(torch.cat([y, agg], dim=-1))

        return h + out

    def forward(self, h: Tensor) -> Tensor:
        if self.checkpointing:
            return checkpoint(self._forward, reentrant=not self.training)(h)
        else:
            return self._forward(h)


class GraphPool(nn.Module):
    r"""Creates a message-passing layer that maps node features from a source level
    onto a (generally smaller or larger) target level via directed edges.

    Used both for pooling (grid -> mesh, fine mesh -> coarse mesh) in the encoder
    and unpooling (coarse mesh -> fine mesh, mesh -> grid) in the decoder; only the
    direction of the supplied edges and the channel counts differ.

    Since source and target levels generally have different node counts, there is no
    true identity map to fall back on. Instead, a linear skip connection (projected
    through the same edges) plays the role GraphResBlock's "h +" residual plays: it
    carries a healthy, un-shrunk gradient path, while the MLP learns a small correction
    on top of it (identity_init shrinks the MLP's output layer, not the skip). Without
    this skip, stacking several GraphPools in a row (as GraphEncoder/GraphDecoder do,
    with no residual connection between them) vanishes gradients by several orders of
    magnitude before they reach the earliest layers.

    Arguments:
        in_channels: Number of source node features C_i.
        out_channels: Number of target node features C_o.
        edge_index: Source-to-target edge index, with shape (2, E).
        num_targets: Number of nodes N_o in the target level.
        ffn_factor: Channel expansion factor in the MLP.
        identity_init: Initialize the MLP's output layer with small residual noise (scale 1e-2).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_index: Tensor,
        num_targets: int,
        ffn_factor: int = 1,
        identity_init: bool = True,
    ) -> None:
        super().__init__()

        self.register_buffer("edge_index", edge_index)
        self.register_buffer("degree", _in_degree(edge_index, num_targets))

        self.num_targets = num_targets

        self.skip = nn.Linear(in_channels, out_channels, bias=False)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, ffn_factor * out_channels),
            nn.SiLU(),
            nn.Linear(ffn_factor * out_channels, out_channels),
        )

        if identity_init:
            self.mlp[-1].weight.data.mul_(1e-2)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, h: Tensor) -> Tensor:
        r"""
        Arguments:
            h: Source node features, with shape (B, N_i, C_i).

        Returns:
            Target node features, with shape (B, N_o, C_o).
        """

        src, dst = self.edge_index

        source = h[:, src]
        messages = self.skip(source) + self.mlp(source)

        out = h.new_zeros(h.shape[0], self.num_targets, messages.shape[-1])
        out = out.index_add_(1, dst, messages) / self.degree

        return out
