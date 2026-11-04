r"""Distributed training utilities."""

__all__ = [
    "reduce_mean",
    "setup_distributed",
]

import os
import torch
import torch.distributed as dist


def reduce_mean(
    value: float,
    device: torch.device,
) -> float:
    r"""Reduce a scalar value across distributed processes by averaging.

    Arguments:
        value  : Scalar value local to the current process.
        device : Device on which the all-reduce is performed.

    Returns:
        value : Value averaged over every process.
    """

    t = torch.tensor(value, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.AVG)

    return t.item()


def setup_distributed() -> tuple[int, int, int, torch.device, bool]:
    r"""Initialize distributed training and return the process configuration.

    Returns:
        rank           : Global rank of the current process.
        local_rank     : Local rank, i.e. GPU index on this node.
        world_size     : Total number of processes across all nodes.
        device         : Torch device assigned to this process.
        is_distributed : True when running under torchrun, False otherwise.
    """

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Running under torchrun
        if "LOCAL_RANK" not in os.environ:
            raise KeyError(
                "ERROR - LOCAL_RANK not found. Ensure training is launched with torchrun."
            )

        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        use_cuda = torch.cuda.is_available()

        # Initialize process group (torchrun sets up env variables)
        dist.init_process_group(backend="nccl" if use_cuda else "gloo", init_method="env://")

        # Set device for this process
        if use_cuda:
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(local_rank)
            torch.set_float32_matmul_precision("high")
        else:
            device = torch.device("cpu")

        return rank, local_rank, world_size, device, True

    else:
        # Single-GPU or CPU fallback
        rank, local_rank, world_size = 0, 0, 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return rank, local_rank, world_size, device, False
