r"""Saving and loading tools for PyTorch models."""

__all__ = [
    "save",
    "load",
    "load_config",
    "load_weights",
]

import torch
import torch.nn as nn

from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Any, Type, Union


def save(
    model: nn.Module,
    config: Union[DictConfig, dict[str, Any]],
    path: Union[str, Path],
) -> None:
    r"""Saves the weights and configuration of a model to a directory.

    Arguments:
        model: Model to save.
        config: Configuration of model.
        path: Target directory in which to save the model.
    """

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(config, path / "config.yml")
    torch.save(model.state_dict(), path / "model.pth")


def load_config(path: Union[str, Path]) -> DictConfig:
    r"""Loads configuration file of a saved model."""
    return OmegaConf.load(Path(path) / "config.yml")


def load_weights(model: nn.Module, path: Union[str, Path], device: str = "cuda") -> nn.Module:
    r"""Loads saved weights into a model.

    Arguments:
        model: Model to load weights into.
        path: Directory containing the saved model weights.
        device: Device to load the model onto (e.g. "cpu", "cuda").

    Returns:
        model: The same model, with the saved weights, on device and in eval mode.
    """

    # Checking for CUDA availability
    device = "cpu" if device == "cuda" and not torch.cuda.is_available() else device

    state = torch.load(Path(path) / "model.pth", map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def load(path: Union[str, Path], model_cls: Type[nn.Module], device: str = "cuda") -> nn.Module:
    r"""Loads a model of a given class from a saved checkpoint.

    Arguments:
        path: Directory containing the saved model checkpoint.
        model_cls: Class to instantiate from the saved configuration.
        device: Device to load the model onto (e.g. "cpu", "cuda").

    Returns:
        model: The reconstructed model, with the saved weights, on device and in eval mode.
    """

    config = load_config(path)
    return load_weights(model_cls(**config), path, device=device)
