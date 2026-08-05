r"""Graph Auto-Encoder (GAE) building blocks.

Architecture inspired by the hierarchical mesh graph neural networks used in

    | Njord: A Probabilistic Graph Neural Network for Ensemble Ocean Forecasting
    | (Holmberg et al., 2026)
    | https://arxiv.org/abs/2605.15470

Unlike a :class:`shaggy.models.cae.ConvAE`, which compresses a regular grid tensor
(B, C, L_1, ..., L_N) by striding convolutions, a :class:`GraphAE` compresses it by
routing grid nodes through a hierarchy of coarser and coarser mesh graphs (built by
:func:`shaggy.models.tools.build_mesh`) with graph neural network message passing. The
mesh geometry is fixed at construction time: a :class:`GraphAE` is only valid for
the input resolution it was built for.
"""

__all__ = [
    "Mesh",
    "GraphEncoder",
    "GraphDecoder",
    "GraphAE",
    "create_GraphAE",
]

import math
import torch.nn as nn

from torch import Tensor
from typing import List, NamedTuple, Optional, Sequence, Tuple

from shaggy.layers import GraphPool, GraphResBlock
from shaggy.models.ae import AutoEncoder


class Mesh(NamedTuple):
    r"""Hierarchical mesh graph consumed by :class:`GraphEncoder` and :class:`GraphDecoder`.

    Level 0 is the input grid itself (one node per pixel/voxel); levels 1 to L are
    increasingly coarse mesh levels, L being :attr:`num_levels`. Built by
    :func:`shaggy.models.tools.build_mesh`.

    Attributes:
        resolution: Shape of the input grid (L_1, ..., L_N).
        pos: Node positions per level, from finest (index 0) to coarsest (index L),
             with shapes (N_0, N), ..., (N_L, N).
        edges: Intra-level edge index per level, with shapes (2, E_0), ..., (2, E_L).
        edges_down: Inter-level edge index per level l -> l + 1 (fine to coarse),
                    with shapes (2, N_0), ..., (2, N_{L-1}).
        edges_up: Inter-level edge index per level l + 1 -> l (coarse to fine),
                  with shapes (2, N_0), ..., (2, N_{L-1}).
        mask: Optional flat boolean mask, with shape (prod(resolution),), marking which
              grid cells are valid (level-0) nodes. None means every grid cell is a node.
              Used for irregular domains (e.g. ocean data with land cells).
    """

    resolution: Tuple[int, ...]
    pos: List[Tensor]
    edges: List[Tensor]
    edges_down: List[Tensor]
    edges_up: List[Tensor]
    mask: Optional[Tensor] = None

    @property
    def num_levels(self) -> int:
        return len(self.pos) - 1


class GraphEncoder(nn.Module):
    r"""Creates a graph encoder.

    Flattens the input grid into nodes, projects them onto the finest mesh level
    (grid-to-mesh pooling), then sweeps upward through the mesh hierarchy, applying
    :class:`GraphResBlock`s at each level and coarsening with :class:`GraphPool`
    between levels.

    Arguments:
        in_channels: Number of input channels C_i.
        out_channels: Number of output (latent) channels C_o.
        mesh: Hierarchical mesh graph, with mesh.num_levels == len(hid_channels).
        hid_channels: Numbers of channels at each mesh level.
        hid_blocks: Numbers of hidden blocks at each mesh level.
        ffn_factor: Channel expansion factor in each MLP.
        dropout: Dropout rate in [0, 1].
        checkpointing: Whether to use gradient checkpointing or not.
        identity_init: Initialize pooling layers' MLP output with small residual noise, see GraphPool.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mesh: Mesh,
        hid_channels: Sequence[int] = (64, 128, 256),
        hid_blocks: Sequence[int] = (3, 3, 3),
        ffn_factor: int = 1,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
        identity_init: bool = True,
    ) -> None:
        super().__init__()

        assert len(hid_blocks) == len(hid_channels)
        assert mesh.num_levels == len(hid_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = mesh.resolution
        self.num_latent_nodes = mesh.pos[-1].shape[0]

        if mesh.mask is not None:
            self.register_buffer("grid_mask", mesh.mask)
        else:
            self.grid_mask = None

        self.ascent = nn.ModuleList()

        for i, num_blocks in enumerate(hid_blocks):
            blocks = nn.ModuleList()

            level = i + 1
            source_channels = in_channels if i == 0 else hid_channels[i - 1]

            blocks.append(
                GraphPool(
                    source_channels,
                    hid_channels[i],
                    edge_index=mesh.edges_down[i],
                    num_targets=mesh.pos[level].shape[0],
                    ffn_factor=ffn_factor,
                    identity_init=identity_init,
                )
            )

            for _ in range(num_blocks):
                blocks.append(
                    GraphResBlock(
                        hid_channels[i],
                        edge_index=mesh.edges[level],
                        num_nodes=mesh.pos[level].shape[0],
                        ffn_factor=ffn_factor,
                        dropout=dropout,
                        checkpointing=checkpointing,
                    )
                )

            if i + 1 == len(hid_blocks):
                blocks.append(nn.Linear(hid_channels[i], out_channels))

            self.ascent.append(blocks)

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: Input tensor, with shape (B, C_i, L_1, ..., L_N). Cells outside the
               mesh's mask (if any) are never read and may hold any value (e.g. NaN).

        Returns:
            z: Latent node features, with shape (B, M, C_o), M being the number of
               nodes at the coarsest mesh level.
        """

        h = x.flatten(2).transpose(1, 2)

        if self.grid_mask is not None:
            h = h[:, self.grid_mask]

        for blocks in self.ascent:
            for block in blocks:
                h = block(h)

        return h


class GraphDecoder(nn.Module):
    r"""Creates a graph decoder.

    Sweeps downward through the mesh hierarchy from the coarsest level, applying
    :class:`GraphResBlock`s at each level and refining with :class:`GraphPool`
    between levels, then unpools the finest mesh level back onto the grid
    (mesh-to-grid unpooling) and reshapes it.

    Arguments:
        in_channels: Number of input (latent) channels C_i.
        out_channels: Number of output channels C_o.
        mesh: Hierarchical mesh graph, with mesh.num_levels == len(hid_channels).
        hid_channels: Numbers of channels at each mesh level.
        hid_blocks: Numbers of hidden blocks at each mesh level.
        ffn_factor: Channel expansion factor in each MLP.
        dropout: Dropout rate in [0, 1].
        checkpointing: Whether to use gradient checkpointing or not.
        identity_init: Initialize unpooling layers' MLP output with small residual noise, see GraphPool.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mesh: Mesh,
        hid_channels: Sequence[int] = (64, 128, 256),
        hid_blocks: Sequence[int] = (3, 3, 3),
        ffn_factor: int = 1,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
        identity_init: bool = True,
    ) -> None:
        super().__init__()

        assert len(hid_blocks) == len(hid_channels)
        assert mesh.num_levels == len(hid_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = mesh.resolution
        self.resolution_numel = math.prod(mesh.resolution)

        if mesh.mask is not None:
            self.register_buffer("grid_mask", mesh.mask)
        else:
            self.grid_mask = None

        self.descent = nn.ModuleList()

        for i, num_blocks in reversed(list(enumerate(hid_blocks))):
            blocks = nn.ModuleList()

            level = i + 1

            if i + 1 == len(hid_blocks):
                blocks.append(nn.Linear(in_channels, hid_channels[i]))

            for _ in range(num_blocks):
                blocks.append(
                    GraphResBlock(
                        hid_channels[i],
                        edge_index=mesh.edges[level],
                        num_nodes=mesh.pos[level].shape[0],
                        ffn_factor=ffn_factor,
                        dropout=dropout,
                        checkpointing=checkpointing,
                    )
                )

            target_channels = hid_channels[i - 1] if i > 0 else out_channels

            blocks.append(
                GraphPool(
                    hid_channels[i],
                    target_channels,
                    edge_index=mesh.edges_up[i],
                    num_targets=mesh.pos[i].shape[0],
                    ffn_factor=ffn_factor,
                    identity_init=identity_init,
                )
            )

            self.descent.append(blocks)

    def forward(self, z: Tensor) -> Tensor:
        r"""
        Arguments:
            z: Latent node features, with shape (B, M, C_i), M being the number of
               nodes at the coarsest mesh level.

        Returns:
            Output tensor, with shape (B, C_o, L_1, ..., L_N). Cells outside the
            mesh's mask (if any) are set to zero.
        """

        h = z

        for blocks in self.descent:
            for block in blocks:
                h = block(h)

        batch_size = h.shape[0]

        if self.grid_mask is not None:
            grid = h.new_zeros(batch_size, self.resolution_numel, self.out_channels)
            grid[:, self.grid_mask] = h
        else:
            grid = h

        x = grid.transpose(1, 2).reshape(batch_size, self.out_channels, *self.resolution)

        return x


class GraphAE(AutoEncoder):
    r"""Creates a graph auto-encoder module.

    Arguments:
        encoder: Encoder module.
        decoder: Decoder module.
        saturation: Saturation function applied to latent codes.
        saturation_bound: Bound used by the saturation function.
    """

    def latent_shape(self) -> Tuple[int, int]:
        r"""Returns the latent node-feature shape for this model's (fixed) mesh.

        Unlike :meth:`shaggy.models.cae.ConvAE.latent_shape`, no resolution argument
        is needed: the mesh, and therefore the input resolution, is fixed once the
        model is built.

        Returns:
            shape: Latent tensor shape (M, C_z), M being the number of nodes at the
                   coarsest mesh level and C_z the number of latent channels.
        """

        return self.encoder.num_latent_nodes, self.encoder.out_channels

    def compression_info(self, input_shape: Sequence[int]) -> Tuple[Tuple[int, int], int]:
        r"""Returns the bottleneck latent shape and compression factor for a given input shape.

        Arguments:
            input_shape: Full input dimensions (C, L_1, ..., L_N), must match the
                         resolution the mesh was built for.

        Returns:
            latent: Latent tensor shape (M, C_z).
            factor: Integer compression factor = prod(input_shape) // prod(latent).
        """

        lat = self.latent_shape()
        factor = math.prod(input_shape) // math.prod(lat)

        return lat, factor


def create_GraphAE(
    in_channels: int,
    out_channels: int,
    lat_channels: int,
    mesh: Mesh,
    saturation_bound: float = 5.0,
    saturation: Optional[str] = "softclip2",
    **kwargs,
) -> GraphAE:
    r"""Instantiates a graph auto-encoder.

    Arguments:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        lat_channels: Number of latent channels.
        mesh: Hierarchical mesh graph built by shaggy.models.tools.build_mesh, with
              mesh.num_levels == len(hid_channels).
        saturation_bound: Bound used by the saturation function.
        saturation: Saturation function applied to latent codes.
        **kwargs: Forwarded to both GraphEncoder and GraphDecoder.

    Returns:
        autoencoder: A GraphAE instance.
    """

    encoder = GraphEncoder(
        in_channels=in_channels,
        out_channels=lat_channels,
        mesh=mesh,
        **kwargs,
    )

    decoder = GraphDecoder(
        in_channels=lat_channels,
        out_channels=out_channels,
        mesh=mesh,
        **kwargs,
    )

    return GraphAE(
        encoder,
        decoder,
        saturation=saturation,
        saturation_bound=saturation_bound,
    )
