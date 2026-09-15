r"""Optimizers sub-package."""

__all__ = [
    "safe_gradient_step",
    "SOAP",
    "Muon",
    "HybridMA",
]

from .gradients import safe_gradient_step
from .soap import SOAP

try:
    from .hybrid import HybridMA
    from .muon import Muon
except AttributeError:
    # Muon subclasses torch.optim.Muon, only available in newer PyTorch releases.
    HybridMA = None
    Muon = None
