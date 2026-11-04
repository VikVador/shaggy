r"""Tests for Shaggy distributed tools."""

import pytest

from shaggy.distributed import setup_distributed


def test_setup_distributed_single_process(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Determines if it falls back to a single process outside of torchrun."""

    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    rank, local_rank, world_size, _, is_distributed = setup_distributed()

    assert (rank, local_rank, world_size, is_distributed) == (0, 0, 1, False)
