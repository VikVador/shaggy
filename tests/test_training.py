r"""Tests for Shaggy training tools: stopping, fixed samples, infinite dataloader, saved states."""

import torch

from pathlib import Path
from torch.utils.data import DataLoader

from shaggy.optimizers.soap import SOAP
from shaggy.schedulers import warmup_cosine_decay
from shaggy.training import (
    get_fixed_indices,
    has_stagnated,
    infinite_dataloader,
    load_state,
    save_state,
)


def test_has_stagnated_flat() -> None:
    r"""Determines if it stops on a flat validation."""

    history = [1.0, 0.5, 0.25] + [0.1, 0.1001, 0.0999, 0.1] * 3

    assert has_stagnated(history, window=10, threshold=0.01)


def test_has_stagnated_falling() -> None:
    r"""Determines if it continues while the validation falls."""

    history = [0.9**i for i in range(30)]

    assert not has_stagnated(history, window=10, threshold=0.01)


def test_has_stagnated_rising() -> None:
    r"""Determines if it stops on a rising validation, whose spread stays large."""

    history = [0.1 * 1.1**i for i in range(20)]

    assert has_stagnated(history, window=10, threshold=0.01)


def test_has_stagnated_too_short() -> None:
    r"""Determines if it waits for a full window, without a previous best to compare with."""

    assert not has_stagnated([1.0, 2.0, 3.0], window=10, threshold=0.01)
    assert not has_stagnated([1.0 + i for i in range(10)], window=10, threshold=0.01)


def test_get_fixed_indices_typical() -> None:
    r"""Determines if the days are always the same, spread through time and mixed."""

    indices = get_fixed_indices(730, 512)

    assert indices == get_fixed_indices(730, 512)
    assert len(set(indices)) == 512
    assert min(indices) == 0 and max(indices) == 729
    assert indices != sorted(indices)
    assert max(b - a for a, b in zip(sorted(indices), sorted(indices)[1:], strict=False)) <= 2


def test_get_fixed_indices_all() -> None:
    r"""Determines if asking for every day keeps each of them once."""

    assert sorted(get_fixed_indices(16, 16)) == list(range(16))


def test_infinite_dataloader_cycles() -> None:
    r"""Determines if it cycles past the end of the dataloader and stops after the requested batches."""

    dataloader = DataLoader(torch.arange(6), batch_size=2)
    batches = [batch.tolist() for batch in infinite_dataloader(dataloader, batches=5)]

    assert batches == [[0, 1], [2, 3], [4, 5], [0, 1], [2, 3]]


def test_state_resumes_exactly(tmp_path: Path) -> None:
    r"""Determines if a training resumed from a saved state follows the same path as if never stopped."""

    def build() -> tuple:
        torch.manual_seed(0)
        model = torch.nn.Linear(8, 4)
        optimizer = SOAP(model.parameters(), lr=1e-2, max_precond_size=1024)
        scheduler = warmup_cosine_decay(optimizer, 1e-3, 1e-2, 1e-4, 2, 40)
        return model, optimizer, scheduler

    def step(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        i: int,
    ) -> None:
        torch.manual_seed(i)
        model(torch.randn(16, 8)).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    model, optimizer, scheduler = build()
    for i in range(20):
        step(model, optimizer, scheduler, i)

    path = tmp_path / "last" / "state.pt"
    save_state(path, {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()})
    weights = {k: v.clone() for k, v in model.state_dict().items()}

    for i in range(20, 30):
        step(model, optimizer, scheduler, i)

    resumed, optimizer_resumed, scheduler_resumed = build()
    state = load_state(path, torch.device("cpu"))
    resumed.load_state_dict(weights)
    optimizer_resumed.load_state_dict(state["optimizer"])
    scheduler_resumed.load_state_dict(state["scheduler"])

    for i in range(20, 30):
        step(resumed, optimizer_resumed, scheduler_resumed, i)

    assert path.exists() and not path.with_suffix(".tmp").exists()
    assert scheduler_resumed.get_last_lr() == scheduler.get_last_lr()
    for key, value in model.state_dict().items():
        assert torch.allclose(value, resumed.state_dict()[key], atol=1e-5)
