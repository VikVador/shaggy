r"""Tools to stop and resume a training."""

__all__ = [
    "has_stagnated",
    "load_state",
    "save_state",
]

import numpy as np
import torch

from collections.abc import Sequence
from pathlib import Path
from typing import Any


def has_stagnated(history: Sequence[float], window: int, threshold: float) -> bool:
    r"""Determine if a loss stopped improving, flat over its last values or above its best value.

    Arguments:
        history   : Losses, from the oldest to the newest.
        window    : Number of last losses considered.
        threshold : Relative standard deviation under which the last losses are flat.

    Returns:
        stagnated : Whether the training can stop.
    """

    # Security
    assert window > 1, f"ERROR - window must be > 1, got {window}."
    assert threshold > 0, f"ERROR - threshold must be > 0, got {threshold}."

    if len(history) < window:
        return False

    last = np.asarray(history[-window:])

    # A rising loss has a large spread, the second test stops it when no recent loss beats the older ones
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
