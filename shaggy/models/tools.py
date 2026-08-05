r"""Mesh construction for GraphAE models."""

__all__ = [
    "build_mesh",
]

import torch

from torch import Tensor
from typing import Optional, Sequence, Tuple

from shaggy.models.gae import Mesh


def _kmeans(
    pos: Tensor,
    num_clusters: int,
    iterations: int = 50,
    axis_scale: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Clusters points with Lloyd's K-means algorithm.

    Arguments:
        pos: Point coordinates, with shape (N, D).
        num_clusters: Number of clusters K.
        iterations: Number of Lloyd iterations.
        axis_scale: Optional per-axis scale, with shape (D,), applied only to the
                    distance used for cluster assignment, not to the returned
                    centroids' coordinates. None means isotropic (all axes count
                    equally).
        generator: Optional random generator for centroid initialization.

    Returns:
        centroids: Cluster centroids, with shape (K, D).
        assignment: Cluster index of each point, with shape (N,).
    """
    indices = torch.randperm(pos.shape[0], generator=generator)[:num_clusters]
    centroids = pos[indices].clone()

    scale = 1.0 if axis_scale is None else axis_scale

    assignment = torch.zeros(pos.shape[0], dtype=torch.long)

    for _ in range(iterations):
        assignment = torch.cdist(pos * scale, centroids * scale).argmin(dim=1)

        for k in range(num_clusters):
            mask = assignment == k
            if mask.any():
                centroids[k] = pos[mask].mean(dim=0)

    return centroids, assignment


def _knn_graph(pos: Tensor, k: int, axis_scale: Optional[Tensor] = None) -> Tensor:
    r"""Builds a symmetric k-nearest-neighbor graph over a set of points.

    Arguments:
        pos: Point coordinates, with shape (N, D).
        k: Number of nearest neighbors per point.
        axis_scale: Optional per-axis scale, with shape (D,), applied to the distance
                    used to find neighbors. None means isotropic (all axes count
                    equally).

    Returns:
        edge_index: Edge index, with shape (2, E).
    """
    n = pos.shape[0]
    k = min(k, n - 1)

    if k <= 0:
        return torch.zeros(2, 0, dtype=torch.long)

    scale = 1.0 if axis_scale is None else axis_scale

    distances = torch.cdist(pos * scale, pos * scale)
    distances.fill_diagonal_(float("inf"))

    neighbors = distances.topk(k, largest=False).indices

    src = torch.arange(n).unsqueeze(1).expand(-1, k).reshape(-1)
    dst = neighbors.reshape(-1)

    edge_index = torch.stack([src, dst], dim=0)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index = torch.unique(edge_index, dim=1)

    return edge_index


def build_mesh(
    resolution: Sequence[int],
    num_levels: int,
    reduction: int = 4,
    k: int = 6,
    iterations: int = 50,
    mask: Optional[Tensor] = None,
    axis_scale: Optional[Sequence[float]] = None,
    seed: Optional[int] = None,
) -> Mesh:
    r"""Builds a hierarchical mesh graph over a regular grid, for shaggy.models.gae.GraphAE.

    Grid nodes are placed at the coordinates of a regular (L_1, ..., L_N) grid,
    normalized to [0, 1]. If `mask` is given, only the True cells become grid nodes
    (e.g. sea cells of an irregular ocean domain, land cells excluded entirely). Each
    mesh level is obtained by K-means clustering of the previous (finer) level's node
    coordinates, reducing the node count by roughly `reduction`. Intra-level edges form
    a symmetric k-nearest-neighbor graph (skipped for the grid level itself, which is
    never used for message passing and can be too large for an all-pairs graph);
    inter-level edges directly reuse the K-means cluster assignment (a fine node is
    connected to the coarse node/cluster it belongs to, and vice versa).

    Arguments:
        resolution: Shape of the input grid (L_1, ..., L_N).
        num_levels: Number of mesh levels L to build on top of the grid level. Must
                    match len(hid_channels) of the GraphEncoder/GraphDecoder.
        reduction: Approximate factor by which the node count shrinks at each level.
        k: Number of neighbors used to build each mesh level's intra-level graph.
        iterations: Number of Lloyd's K-means iterations used at each level.
        mask: Optional boolean tensor, with shape (L_1, ..., L_N), marking valid
              (e.g. sea) grid cells. None means every grid cell is a node.
        axis_scale: Optional per-axis scale (s_1, ..., s_N), one per resolution
                    dimension. Rescales node coordinates only for the K-means/k-NN
                    distance computations (not the node positions themselves): a
                    larger scale on an axis makes it dominate the distance, so
                    neighbors/clusters are chosen mainly for closeness along that
                    axis (tight packing along it, looser along the others); a
                    smaller scale makes it barely count, so neighbors are chosen
                    almost freely along it (wider reach along that axis, tighter
                    packing along the others). This is how to bias mesh connectivity
                    towards (or away from) a given axis, e.g. a depth/Z axis if
                    `resolution` grows past (X, Y) — a literal separate neighbor
                    count per axis has no natural meaning for K-means/k-NN over an
                    irregular, masked point cloud, but scaling achieves an
                    equivalent, continuous control. None means isotropic (all axes
                    count equally), matching a plain, unscaled build.
        seed: Optional random seed, for reproducible mesh construction.

    Returns:
        mesh: A Mesh with num_levels levels on top of the grid level.
    """
    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    if axis_scale is None:
        scale = None
    else:
        assert len(axis_scale) == len(resolution)
        scale = torch.as_tensor(axis_scale, dtype=torch.float32)

    axes = [torch.linspace(0.0, 1.0, r) for r in resolution]
    grid = torch.meshgrid(*axes, indexing="ij")
    grid_pos = torch.stack(grid, dim=-1).reshape(-1, len(resolution))

    if mask is None:
        mask_flat = None
    else:
        mask_flat = mask.reshape(-1).bool()
        grid_pos = grid_pos[mask_flat]

    pos = [grid_pos]

    edges_down, edges_up = [], []

    for _ in range(num_levels):
        num_clusters = max(1, pos[-1].shape[0] // reduction)

        centroids, assignment = _kmeans(
            pos[-1],
            num_clusters,
            iterations=iterations,
            axis_scale=scale,
            generator=generator,
        )

        fine = torch.arange(pos[-1].shape[0])

        edges_down.append(torch.stack([fine, assignment], dim=0))
        edges_up.append(torch.stack([assignment, fine], dim=0))

        pos.append(centroids)

    # The grid level is never used for intra-level message passing, and can be far too
    # large for an all-pairs k-NN graph (e.g. tens of thousands of ocean grid cells).
    edges = [torch.zeros(2, 0, dtype=torch.long)] + [
        _knn_graph(p, k, axis_scale=scale) for p in pos[1:]
    ]

    return Mesh(
        resolution=tuple(resolution),
        pos=pos,
        edges=edges,
        edges_down=edges_down,
        edges_up=edges_up,
        mask=mask_flat,
    )
