r"""Neural network utils."""

import torch
import torch.utils.checkpoint

from torch._C._functorch import is_gradtrackingtensor
from typing import Any, Callable, Optional, Tuple


class CheckpointReentrant(torch.autograd.Function):
    r"""Reentrant activation checkpointing compatible with both backward and forward AD.

    Unlike torch.utils.checkpoint.checkpoint, this implementation supports forward-mode
    AD (jvp) in addition to reverse-mode AD (vjp). The trade-off is that gradients
    are only propagated to the explicit positional tensor inputs of the wrapped function;
    implicit inputs such as module parameters do not receive gradients.
    """

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple, outputs: Any) -> None:
        r"""Stores detached inputs and the wrapped function for recomputation."""
        func, *xs = inputs

        xs = [x.detach() for x in xs]

        ctx.save_for_backward(*xs)
        ctx.save_for_forward(*xs)
        ctx.func = func

    @staticmethod
    def forward(func: Callable, *xs: torch.Tensor) -> Any:
        r"""Runs the wrapped function in the forward pass."""
        return func(*xs)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def vjp(ctx: Any, *grad_ys: torch.Tensor) -> Tuple:
        r"""Recomputes the forward graph and applies reverse-mode AD."""
        xs = ctx.saved_tensors

        with torch.enable_grad():
            xs = [x.detach().requires_grad_() for x in xs]
            ys = ctx.func(*xs)

        if torch.is_tensor(ys):
            ys = [ys]

        grad_xs = torch.autograd.grad(ys, xs, grad_ys)

        return (None, *grad_xs)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def jvp(ctx: Any, grad_func: torch.Tensor, *grad_xs: torch.Tensor) -> torch.Tensor:
        r"""Applies forward-mode AD via torch.func.jvp."""
        xs = ctx.saved_tensors

        _, grad_ys = torch.func.jvp(ctx.func, xs, grad_xs)

        return grad_ys


def checkpoint(f: Callable, reentrant: bool = False) -> Callable:
    r"""Applies activation checkpointing to a function.

    Activation checkpointing reduces memory consumption by storing the inputs of the
    function and recomputing its graph during automatic differentiation (AD).

    Reentrant checkpointing is compatible with backward and forward AD, but only
    propagates gradients to the explicit positional inputs of the function. Implicit
    inputs, such as module parameters, do not get gradients. Conversely, non-reentrant
    will propagate gradients to implicit inputs, but is not compatible with forward AD.

    Arguments:
        f: A function.
        reentrant: Whether to use reentrant checkpointing or not.

    Returns:
        The checkpointed function.
    """

    def g(*args: Any, **kwargs: Any) -> Any:
        mask = [
            torch.is_tensor(arg)
            and torch.is_floating_point(arg)
            and (arg.requires_grad or is_gradtrackingtensor(arg))
            for arg in args
        ]

        tensors = [arg for include, arg in zip(mask, args, strict=True) if include]
        others = [arg for include, arg in zip(mask, args, strict=True) if not include]

        def h(*tensors: torch.Tensor) -> Any:
            it, io = iter(tensors), iter(others)
            args = (next(it if include else io) for include in mask)
            return f(*args, **kwargs)

        if reentrant:
            if tensors:
                return CheckpointReentrant.apply(h, *tensors)
            else:
                with torch.no_grad():
                    return h(*tensors)
        else:
            return torch.utils.checkpoint.checkpoint(h, *tensors, use_reentrant=False)

    return g


class skip_init(torch.overrides.TorchFunctionMode):
    r"""Context that skips all weight initialization calls from torch.nn.init.

    Useful when building large models where random initialization is expensive and
    weights will be immediately overwritten (e.g. loaded from a checkpoint).
    """

    def __torch_function__(
        self, func: Callable, types: Any, args: Tuple = (), kwargs: Optional[dict] = None
    ) -> Any:
        r"""Intercepts torch.nn.init.* calls and returns the tensor unchanged."""
        kwargs = kwargs or {}
        if getattr(func, "__module__", None) == "torch.nn.init":
            if "tensor" in kwargs:
                return kwargs["tensor"]
            else:
                return args[0]
        else:
            return func(*args, **kwargs)
