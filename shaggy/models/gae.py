r"""Graph Auto-Encoder (GAE) building blocks.

Architecture inspired by the hierarchical mesh graph neural networks used in

    | Njord: A Probabilistic Graph Neural Network for Ensemble Ocean Forecasting
    | (Holmberg et al., 2026)
    | https://arxiv.org/abs/2605.15470

Unlike a :class:`shaggy.models.cae.ConvAE`, which compresses a regular grid tensor
(B, C, L_1, ..., L_N) by striding convolutions, a :class:`GraphAE` compresses it by
routing grid nodes through a hierarchy of coarser and coarser mesh graphs (built by
:func:`shaggy.tools.build_mesh`) with graph neural network message passing. The
mesh geometry is fixed at construction time: a :class:`GraphAE` is only valid for
the input resolution it was built for.
"""

__all__ = [
    "Mesh",
    "GraphResBlock",
    "GraphPool",
    "GraphEncoder",
    "GraphDecoder",
    "GraphAE",
    "create_GraphAE",
]

import math
import torch
import torch.nn as nn

from azula.nn.utils import get_module_dtype
from torch import Tensor
from typing import List, NamedTuple, Optional, Sequence, Tuple

from shaggy.layers import LayerNorm
from shaggy.utils import checkpoint


class Mesh(NamedTuple):
    r"""Hierarchical mesh graph consumed by :class:`GraphEncoder` and :class:`GraphDecoder`.

    Level 0 is the input grid itself (one node per pixel/voxel); levels 1 to L are
    increasingly coarse mesh levels, L being :attr:`num_levels`. Built by
    :func:`shaggy.tools.build_mesh`.

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

    Analogous to :class:`shaggy.models.cae.ResBlock`, but the spatial convolution is
    replaced by an edge-conditioned message passing step (an Interaction Network,
    following Njord/GraphCast-style GNNs) over a fixed graph.

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


class GraphAE(nn.Module):
    r"""Creates a graph auto-encoder module.

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
            return x * torch.rsqrt(torch.mean(torch.square(x), dim=-1, keepdim=True) + 1e-5)
        else:
            raise ValueError(f"unknown saturation '{self.saturation}'")

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

    def encode(self, x: Tensor) -> Tensor:
        r"""Encodes an image tensor into a latent representation.

        Arguments:
            x: Input image, with shape (B, C_i, L_1, ..., L_N).

        Returns:
            z: Latent node features, with shape (B, M, C_z).
        """

        dtype = get_module_dtype(self.encoder)
        z = self.encoder(x.to(dtype))
        z = self.saturate(z)

        return z.to(x.dtype)

    def decode(self, z: Tensor) -> Tensor:
        r"""Decodes a latent code back into an image tensor.

        Arguments:
            z: Latent node features, with shape (B, M, C_z).

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
            z: Latent node features, with shape (B, M, C_z).
            y: Reconstructed image, with shape (B, C_o, L_1, ..., L_N).
        """

        z = self.encode(x)
        y = self.decode(z)

        return z, y


def create_GraphAE(
    in_channels: int,
    out_channels: int,
    lat_channels: int,
    mesh: Mesh,
    noise_level: float = 0.0,
    saturation_bound: float = 5.0,
    saturation: Optional[str] = "softclip2",
    **kwargs,
) -> GraphAE:
    r"""Instantiates a graph auto-encoder.

    Arguments:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        lat_channels: Number of latent channels.
        mesh: Hierarchical mesh graph built by shaggy.tools.build_mesh, with
              mesh.num_levels == len(hid_channels).
        noise_level: Standard deviation of Gaussian noise injected at decode time.
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
        noise=noise_level,
    )
