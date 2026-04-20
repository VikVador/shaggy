r"""Loss for auto-encoder training."""

__all__ = [
    "AELoss",
]

import torch.nn as nn

from torch import Tensor
from typing import Optional


class AELoss(nn.Module):
    r"""Mean squared error loss for auto-encoder training.

    Arguments:
        weights: Optional weight tensor, broadcastable to (B, C_o, L_1, ..., L_N),
                 applied element-wise to squared errors before averaging.
    """

    def __init__(self, weights: Optional[Tensor] = None) -> None:
        super().__init__()

        self.mse = nn.MSELoss(reduction="none")
        self.register_buffer("weights", weights)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        r"""Computes the (optionally weighted) MSE between input and target.

        Arguments:
            input: Predicted image, with shape (B, C_i, L_1, ..., L_N).
            target: Target image, with shape (B, C_o, L_1, ..., L_N).

        Returns:
            loss: Scalar mean squared error.
        """

        loss = self.mse(input, target)

        if self.weights is not None:
            loss = loss * self.weights

        return loss.mean()
