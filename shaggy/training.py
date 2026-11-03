r"""Tools to sample, stop and resume a training."""

__all__ = [
    "get_fixed_indices",
    "has_stagnated",
    "infinite_dataloader",
    "load_state",
    "save_state",
]

import numpy as np
import torch

from collections.abc import Iterator, Sequence
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Any


def has_stagnated(history: Sequence[float], window: int, threshold: float) -> bool:
    r"""Determine if a validation loss stopped improving, flat over its last values or above its best value.

    Arguments:
        history   : Validation losses, from the oldest to the newest.
        window    : Number of last validations considered.
        threshold : Relative standard deviation under which the last validations are flat.

    Returns:
        stagnated : Whether the training can stop.
    """

    # Security
    assert window > 1, f"ERROR - window must be > 1, got {window}."
    assert threshold > 0, f"ERROR - threshold must be > 0, got {threshold}."

    if len(history) < window:
        return False

    last = np.asarray(history[-window:])

    # A rising loss has a large spread, the second test stops it when no recent validation beats the older ones
    is_flat = bool(last.std() / last.mean() < threshold)
    no_progress = len(history) > window and bool(last.min() >= min(history[:-window]))

    return is_flat or no_progress


def save_state(path: Path, state: dict[str, Any]) -> None:
    r"""Save the state of a training, without ever leaving a partial file if the job stops meanwhile.

    Arguments:
        path  : Target file.
        state : State of the training.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.rename(path)


def load_state(path: Path, device: torch.device) -> dict[str, Any]:
    r"""Load the state of a training.

    Arguments:
        path   : File of the state.
        device : Target device of the tensors.

    Returns:
        state : State of the training.
    """

    return torch.load(path, map_location=device, weights_only=False)


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
    r"""Choose samples spread evenly through time, in an order mixing the seasons within each batch.

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
