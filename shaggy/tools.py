r"""Save and load tools for ConvAE models, and mesh construction for GraphAE models."""

__all__ = [
    "save",
    "load",
    "build_mesh",
]

import torch

from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from torch import Tensor
from typing import Optional, Sequence, Tuple, Union

from shaggy.models.cae import ConvAE, create_ConvAE
from shaggy.models.gae import Mesh
from shaggy.utils import skip_init


def save(model: ConvAE, config: DictConfig, path: Union[str, Path]) -> None:
    r"""Saves a ConvAE model weights and configuration to a directory.

    Creates path if it does not exist, then writes config.yml (the OmegaConf
    configuration) and model.pth (the model state dict).

    Arguments:
        model: The ConvAE model to save.
        config: OmegaConf config holding the create_ConvAE keyword arguments.
        path: Target directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(config, path / "config.yml")
    torch.save(model.state_dict(), path / "model.pth")


def load(path: Union[str, Path], device: str = "cpu") -> ConvAE:
    r"""Loads a ConvAE model from a directory.

    Arguments:
        path: Directory containing config.yml and model.pth.
        device: Device to load the model onto (e.g. "cpu", "cuda").

    Returns:
        model: The loaded ConvAE in eval mode.
    """
    path = Path(path)

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    config = OmegaConf.load(path / "config.yml")

    with skip_init():
        model = create_ConvAE(**config)

    state = torch.load(path / "model.pth", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)

    return model.eval()


def _kmeans(
    pos: Tensor,
    num_clusters: int,
    iterations: int = 50,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Clusters points with Lloyd's K-means algorithm.

    Arguments:
        pos: Point coordinates, with shape (N, D).
        num_clusters: Number of clusters K.
        iterations: Number of Lloyd iterations.
        generator: Optional random generator for centroid initialization.

    Returns:
        centroids: Cluster centroids, with shape (K, D).
        assignment: Cluster index of each point, with shape (N,).
    """
    indices = torch.randperm(pos.shape[0], generator=generator)[:num_clusters]
    centroids = pos[indices].clone()

    assignment = torch.zeros(pos.shape[0], dtype=torch.long)

    for _ in range(iterations):
        assignment = torch.cdist(pos, centroids).argmin(dim=1)

        for k in range(num_clusters):
            mask = assignment == k
            if mask.any():
                centroids[k] = pos[mask].mean(dim=0)

    return centroids, assignment


def _knn_graph(pos: Tensor, k: int) -> Tensor:
    r"""Builds a symmetric k-nearest-neighbor graph over a set of points.

    Arguments:
        pos: Point coordinates, with shape (N, D).
        k: Number of nearest neighbors per point.

    Returns:
        edge_index: Edge index, with shape (2, E).
    """
    n = pos.shape[0]
    k = min(k, n - 1)

    if k <= 0:
        return torch.zeros(2, 0, dtype=torch.long)

    distances = torch.cdist(pos, pos)
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
        seed: Optional random seed, for reproducible mesh construction.

    Returns:
        mesh: A Mesh with num_levels levels on top of the grid level.
    """
    generator = torch.Generator().manual_seed(seed) if seed is not None else None

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
            pos[-1], num_clusters, iterations=iterations, generator=generator
        )

        fine = torch.arange(pos[-1].shape[0])

        edges_down.append(torch.stack([fine, assignment], dim=0))
        edges_up.append(torch.stack([assignment, fine], dim=0))

        pos.append(centroids)

    # The grid level is never used for intra-level message passing, and can be far too
    # large for an all-pairs k-NN graph (e.g. tens of thousands of ocean grid cells).
    edges = [torch.zeros(2, 0, dtype=torch.long)] + [_knn_graph(p, k) for p in pos[1:]]

    return Mesh(
        resolution=tuple(resolution),
        pos=pos,
        edges=edges,
        edges_down=edges_down,
        edges_up=edges_up,
        mask=mask_flat,
    )
