r"""Muon optimizer."""

__all__ = [
    "Muon",
]

import torch

from typing import Iterable


class Muon(torch.optim.Muon):
    r"""
    References:
        | Muon: An optimizer for hidden layers in neural networks (Jordan, 2024)
        | https://kellerjordan.github.io/posts/muon/

    Note:
        Only supports 2D parameters.

    Arguments:
        params: Network parameters.
        lr: Learning rate.
        weight_decay: Weight decay coefficient [0, 1].
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        weight_decay: float = 0.1,
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            weight_decay=weight_decay,
            adjust_lr_fn="match_rms_adamw",
        )

        # Security
        assert all(p.ndim == 2 for p in params), "ERROR - Muon only supports 2D parameters."
