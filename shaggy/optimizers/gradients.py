r"""Tools for gradient steps."""

__all__ = [
    "safe_gradient_step",
]

import torch

from torch import (
    Tensor,
    nn,
)
from typing import Optional


def safe_gradient_step(
    optimizer: torch.optim.Optimizer,
    grad_clip: Optional[float] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Tensor:
    r"""Applies a gradient descent optimization step.

    Arguments:
        optimizer: An optimizer.
        grad_clip: Maximum gradient norm.
        scaler: A gradient scaler for automatic mixed precision training.

    Returns:
        Global gradient norm (before clipping).
    """

    if scaler:
        scaler.unscale_(optimizer)

    params = [p for group in optimizer.param_groups for p in group["params"]]

    if grad_clip is None:
        norm = torch.linalg.vector_norm(
            torch.stack([
                torch.linalg.vector_norm(p.grad) for p in params if torch.is_tensor(p.grad)
            ])
        )
    else:
        norm = nn.utils.clip_grad_norm_(params, grad_clip)

    if scaler:
        scaler.step(optimizer)
        scaler.update()
    elif norm.isfinite():
        optimizer.step()

    optimizer.zero_grad()

    return norm
