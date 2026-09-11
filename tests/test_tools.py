r"""Tests for Shaggy tools: save, load_config, load_weights."""

import pytest
import torch
import torch.nn as nn

from omegaconf import OmegaConf
from pathlib import Path

from shaggy.models.cae import create_ConvAE
from shaggy.tools import load_config, load_weights, save

CONFIG = {
    "in_channels": 2,
    "out_channels": 2,
    "lat_channels": 4,
    "hid_channels": [8],
    "hid_blocks": [1],
    "spatial": 2,
}


def test_save_writes_config_and_weights(tmp_path: Path) -> None:
    r"""Determines if save writes config.yml and model.pth in a directory it creates."""
    target = tmp_path / "nested" / "run"

    save(create_ConvAE(**CONFIG), CONFIG, target)

    assert (target / "config.yml").is_file()
    assert (target / "model.pth").is_file()


@pytest.mark.parametrize("as_omegaconf", [False, True])
def test_load_config_roundtrip(tmp_path: Path, as_omegaconf: bool) -> None:
    r"""Determines if load_config returns the saved config, given as a dict or an OmegaConf."""
    config = OmegaConf.create(CONFIG) if as_omegaconf else CONFIG

    save(create_ConvAE(**CONFIG), config, tmp_path)

    assert OmegaConf.to_container(load_config(tmp_path)) == CONFIG


def test_load_weights_roundtrip(tmp_path: Path) -> None:
    r"""Determines if a model rebuilt from the saved config recovers the saved weights exactly."""
    model = create_ConvAE(**CONFIG)
    save(model, CONFIG, tmp_path)

    loaded = load_weights(create_ConvAE(**load_config(tmp_path)), tmp_path)

    for (name, p), p_loaded in zip(model.state_dict().items(), loaded.state_dict().values()):
        assert torch.equal(p, p_loaded), f"weight mismatch for '{name}'"


def test_load_weights_in_place_and_eval(tmp_path: Path) -> None:
    r"""Determines if load_weights fills the given model and returns it in eval mode."""
    save(create_ConvAE(**CONFIG), CONFIG, tmp_path)

    model = create_ConvAE(**CONFIG).train()
    loaded = load_weights(model, tmp_path)

    assert loaded is model
    assert not loaded.training


def test_load_weights_any_module(tmp_path: Path) -> None:
    r"""Determines if the tools work for any nn.Module, not only autoencoders."""
    model = nn.Linear(3, 5)
    save(model, {"in_features": 3, "out_features": 5}, tmp_path)

    loaded = load_weights(nn.Linear(**load_config(tmp_path)), tmp_path)

    assert torch.equal(loaded.weight, model.weight)
    assert torch.equal(loaded.bias, model.bias)


def test_load_weights_architecture_mismatch(tmp_path: Path) -> None:
    r"""Determines if loading weights into a model of another architecture is rejected."""
    save(create_ConvAE(**CONFIG), CONFIG, tmp_path)

    wider = create_ConvAE(**{**CONFIG, "hid_channels": [16]})

    with pytest.raises(RuntimeError):
        load_weights(wider, tmp_path)
