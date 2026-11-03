r"""Tests for Shaggy learning rate schedulers."""

import pytest
import torch

from shaggy.schedulers import warmup_cosine_decay


def test_warmup_cosine_decay_typical() -> None:
    r"""Determines if it starts, peaks and ends at the requested learning rates."""

    optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1e-2)
    scheduler = warmup_cosine_decay(optimizer, 1e-3, 1e-2, 1e-4, warmup_steps=4, total_steps=20)

    rates = []
    for _ in range(21):
        rates.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()

    assert rates[0] == pytest.approx(1e-3)
    assert rates[4] == pytest.approx(1e-2)
    assert rates[20] == pytest.approx(1e-4)
    assert max(rates) == pytest.approx(1e-2)


def test_warmup_cosine_decay_no_warmup() -> None:
    r"""Determines if it starts directly at the peak when there is no warmup."""

    optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1e-2)
    scheduler = warmup_cosine_decay(optimizer, 1e-3, 1e-2, 1e-4, warmup_steps=0, total_steps=10)

    assert scheduler.get_last_lr()[0] == pytest.approx(1e-2)
