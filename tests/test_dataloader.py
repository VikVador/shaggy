r"""Tests for Shaggy dataloader tools: fixed samples and infinite dataloader."""

import torch

from torch.utils.data import DataLoader

from shaggy.dataloader import get_fixed_indices, infinite_dataloader


def test_get_fixed_indices_typical() -> None:
    r"""Determines if the samples are always the same, spread evenly and shuffled."""

    indices = get_fixed_indices(730, 512)

    assert indices == get_fixed_indices(730, 512)
    assert len(set(indices)) == 512
    assert min(indices) == 0 and max(indices) == 729
    assert indices != sorted(indices)
    assert max(b - a for a, b in zip(sorted(indices), sorted(indices)[1:], strict=False)) <= 2


def test_get_fixed_indices_all() -> None:
    r"""Determines if asking for every sample keeps each of them once."""

    assert sorted(get_fixed_indices(16, 16)) == list(range(16))


def test_infinite_dataloader_cycles() -> None:
    r"""Determines if it cycles past the end of the dataloader and stops after the requested batches."""

    dataloader = DataLoader(torch.arange(6), batch_size=2)
    batches = [batch.tolist() for batch in infinite_dataloader(dataloader, batches=5)]

    assert batches == [[0, 1], [2, 3], [4, 5], [0, 1], [2, 3]]
