r"""Tests for Shaggy training tools: stopping and saved states."""

import torch

from pathlib import Path

from shaggy.optimizers.soap import SOAP
from shaggy.schedulers import warmup_cosine_decay
from shaggy.training import has_stagnated, load_state, save_state


def test_has_stagnated_flat() -> None:
    r"""Determines if it stops on a flat loss."""

    history = [1.0, 0.5, 0.25] + [0.1, 0.1001, 0.0999, 0.1] * 3

    assert has_stagnated(history, window=10, threshold=0.01)


def test_has_stagnated_falling() -> None:
    r"""Determines if it continues while the loss falls."""

    history = [0.9**i for i in range(30)]

    assert not has_stagnated(history, window=10, threshold=0.01)


def test_has_stagnated_rising() -> None:
    r"""Determines if it stops on a rising loss, whose spread stays large."""

    history = [0.1 * 1.1**i for i in range(20)]

    assert has_stagnated(history, window=10, threshold=0.01)


def test_has_stagnated_too_short() -> None:
    r"""Determines if it waits for a full window, without a previous best to compare with."""

    assert not has_stagnated([1.0, 2.0, 3.0], window=10, threshold=0.01)
    assert not has_stagnated([1.0 + i for i in range(10)], window=10, threshold=0.01)


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
