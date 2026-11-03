r"""Learning rate schedulers."""

__all__ = [
    "warmup_cosine_decay",
]


import math
import torch


def warmup_cosine_decay(
    optimizer: torch.optim.Optimizer,
    lr_start: float,
    lr_peak: float,
    lr_end: float,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    r"""Build a linear warmup followed by cosine decay learning rate scheduler.

    Arguments:
        optimizer    : Optimizer whose learning rate will be scheduled.
        lr_start     : Learning rate at step 0 (start of warmup).
        lr_peak      : Peak learning rate, reached at the end of warmup.
        lr_end       : Final learning rate at the last training step.
        warmup_steps : Number of steps to linearly ramp from lr_start to lr_peak.
        total_steps  : Total number of optimizer steps (warmup + decay combined).

    Returns:
        A learning rate scheduler.
    """

    # Security
    assert lr_peak > 0, f"ERROR - lr_peak must be > 0, got {lr_peak}."
    assert total_steps > 0, f"ERROR - total_steps must be > 0, got {total_steps}."
    warmup_valid = 0 <= warmup_steps <= total_steps
    assert warmup_valid, f"ERROR - warmup_steps must be in [0, {total_steps}], got {warmup_steps}."

    def _lr_lambda(step: int) -> float:
        r"""Compute the learning rate multiplier for a given optimizer step."""

        # Linear warmup: interpolate from lr_start to lr_peak
        if step < warmup_steps:
            alpha = step / warmup_steps
            return lr_start / lr_peak + (1.0 - lr_start / lr_peak) * alpha

        # Cosine decay: smoothly anneal from lr_peak down to lr_end
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        return lr_end / lr_peak + (1.0 - lr_end / lr_peak) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
