r"""Tools to iterate over and sample from a dataloader."""

__all__ = [
    "get_fixed_indices",
    "infinite_dataloader",
]

import numpy as np

from collections.abc import Iterator
from torch.utils.data import DataLoader


def infinite_dataloader(
    dataloader: DataLoader,
    batches: int,
    is_distributed: bool = False,
) -> Iterator:
    r"""Make a basic PyTorch dataloader 'infinite' by cycling over it.

    Arguments:
        dataloader     : Dataloader to iterate over.
        batches        : Maximum number of batches to yield before stopping.
        is_distributed : Whether running in distributed mode (updates the sampler epoch each cycle).

    Returns:
        batch : Successive batches, cycling past the end of the dataloader as many times as needed.
    """

    epoch = 0
    batches_remaining = batches

    while batches_remaining > 0:
        if is_distributed and hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)

        for batch in dataloader:
            yield batch
            batches_remaining -= 1
            if batches_remaining <= 0:
                return

        epoch += 1


def get_fixed_indices(length: int, samples: int, seed: int = 0) -> list[int]:
    r"""Choose samples spread evenly over a dataset, in a shuffled order.

    Arguments:
        length  : Number of samples of the dataset.
        samples : Number of samples to keep.
        seed    : Seed of the order of the samples.

    Returns:
        indices : Indices of the samples, always the same for a given length, number of samples and seed.
    """

    # Security
    assert 0 < samples <= length, f"ERROR - samples must be in [1, {length}], got {samples}."

    indices = np.linspace(0, length - 1, samples).round().astype(int)
    return np.random.default_rng(seed).permutation(indices).tolist()
