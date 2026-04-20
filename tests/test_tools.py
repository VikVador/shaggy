r"""Tests for shaggy.tools."""

import torch

from omegaconf import OmegaConf
from pathlib import Path

from shaggy.shaggy.models.cae import create_ConvAE
from shaggy.shaggy.tools import load, save

MINIMAL_CONFIG = {
    "in_channels": 2,
    "out_channels": 2,
    "lat_channels": 4,
    "hid_channels": [8],
    "hid_blocks": [1],
    "spatial": 2,
}


def test_tools_save_load_roundtrip(tmp_path: Path) -> None:
    r"""Checks that save/load preserves model weights exactly."""
    config = OmegaConf.create(MINIMAL_CONFIG)
    model = create_ConvAE(**config)

    save(model, config, tmp_path / "run")

    assert (tmp_path / "run" / "config.yml").exists()
    assert (tmp_path / "run" / "model.pth").exists()

    loaded = load(tmp_path / "run", device="cpu")

    for (name, p), (_, p_loaded) in zip(
        model.state_dict().items(),
        loaded.state_dict().items(),
    ):
        assert torch.equal(p, p_loaded), f"weight mismatch for '{name}'"


def test_tools_load_sets_eval_mode(tmp_path: Path) -> None:
    r"""Checks that load always returns the model in eval mode."""
    config = OmegaConf.create(MINIMAL_CONFIG)
    model = create_ConvAE(**config).train()

    save(model, config, tmp_path / "run")

    loaded = load(tmp_path / "run", device="cpu")

    assert not loaded.training


def test_tools_save_creates_directory(tmp_path: Path) -> None:
    r"""Checks that save creates the target directory if it does not exist."""
    config = OmegaConf.create(MINIMAL_CONFIG)
    model = create_ConvAE(**config)
    target = tmp_path / "nested" / "dir"

    save(model, config, target)

    assert target.is_dir()
