r"""Save and load tools for ConvAE models."""

__all__ = [
    "save",
    "load",
]

import torch

from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Union

from shaggy.shaggy.models.cae import ConvAE, create_ConvAE
from shaggy.shaggy.utils import skip_init


def save(model: ConvAE, config: DictConfig, path: Union[str, Path]) -> None:
    r"""Saves a ConvAE model weights and configuration to a directory.

    Creates path if it does not exist, then writes config.yml (the OmegaConf
    configuration) and model.pth (the model state dict).

    Arguments:
        model: The ConvAE model to save.
        config: OmegaConf config holding the create_ConvAE keyword arguments.
        path: Target directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(config, path / "config.yml")
    torch.save(model.state_dict(), path / "model.pth")


def load(path: Union[str, Path], device: str = "cpu") -> ConvAE:
    r"""Loads a ConvAE model from a directory.

    Arguments:
        path: Directory containing config.yml and model.pth.
        device: Device to load the model onto (e.g. "cpu", "cuda").

    Returns:
        model: The loaded ConvAE in eval mode.
    """
    path = Path(path)

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    config = OmegaConf.load(path / "config.yml")

    with skip_init():
        model = create_ConvAE(**config)

    state = torch.load(path / "model.pth", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)

    return model.eval()
