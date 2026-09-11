r"""Hybrid optimizer."""

__all__ = [
    "HybridMA",
]

import torch

from torch import nn
from typing import (
    Any,
    Optional,
)

from .muon import Muon


class HybridMA:
    r"""A hybrid optimizer that routes parameters to Muon or AdamW.

    References:
        | Decoupled Weight Decay Regularization (Loshchilov & Hutter, 2019)
        | https://arxiv.org/abs/1711.05101
        | Muon: An optimizer for hidden layers in neural networks (Jordan, 2024)
        | https://kellerjordan.github.io/posts/muon/

    Arguments:
        model: Neural network whose parameters are optimized.
        lr: Learning rate.
        weight_decay: Weight decay coefficient [0, 1].
        config_muon: Additional keyword arguments passed to Muon.
        config_adamw: Additional keyword arguments passed to AdamW.
        exclude_names: Parameter names to route to AdamW regardless of their shape.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        config_muon: Optional[dict[str, Any]] = None,
        config_adamw: Optional[dict[str, Any]] = None,
        exclude_names: Optional[list[str]] = None,
    ) -> None:
        excluded = set(exclude_names or [])
        muon_params, adamw_params = [], []

        for name, param in model.named_parameters():
            if param.ndim == 2 and name not in excluded:
                muon_params.append(param)
            else:
                adamw_params.append(param)

        muon_kwargs = {"lr": lr, "weight_decay": weight_decay, **(config_muon or {})}
        adamw_kwargs = {"lr": lr, "weight_decay": weight_decay, **(config_adamw or {})}

        # Skip a sub-optimizer entirely if it has no parameters to optimize
        self.opt_muon = Muon(muon_params, **muon_kwargs) if muon_params else None
        self.opt_adamw = torch.optim.AdamW(adamw_params, **adamw_kwargs) if adamw_params else None

    @property
    def param_groups(self) -> list:
        return [
            group
            for opt in (self.opt_muon, self.opt_adamw)
            if opt is not None
            for group in opt.param_groups
        ]

    def step(self) -> None:
        r"""Performs a single optimization step for both sub-optimizers."""
        if self.opt_muon is not None:
            self.opt_muon.step()
        if self.opt_adamw is not None:
            self.opt_adamw.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        r"""Zeroes the gradients of all parameters.

        Arguments:
            set_to_none: If True, sets gradients to None instead of zero.
        """
        if self.opt_muon is not None:
            self.opt_muon.zero_grad(set_to_none=set_to_none)
        if self.opt_adamw is not None:
            self.opt_adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        r"""Returns the optimizer state as a dictionary.

        Returns:
            state: A dict with keys for "muon" and "adamw" optimizers.
        """
        return {
            "muon": self.opt_muon.state_dict() if self.opt_muon is not None else None,
            "adamw": self.opt_adamw.state_dict() if self.opt_adamw is not None else None,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        r"""Loads a previously saved optimizer state.

        Arguments:
            state_dict: A dict as returned by state_dict(), with keys "muon" and "adamw".
        """
        if self.opt_muon is not None:
            self.opt_muon.load_state_dict(state_dict["muon"])
        if self.opt_adamw is not None:
            self.opt_adamw.load_state_dict(state_dict["adamw"])
