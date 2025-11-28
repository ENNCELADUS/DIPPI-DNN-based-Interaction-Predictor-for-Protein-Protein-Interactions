"""
Minimal DDP init/teardown for torchrun (MVP).

Provides pure bootstrap/cleanup for torch.distributed; no logging, no
checkpointing, no early stopping. The orchestrator (run.py) owns DDP
wrapping, model placement, data sampler creation, and all training/eval logic.

References:
- https://docs.pytorch.org/tutorials/beginner/ddp_series_multigpu.html
- https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html
- https://docs.pytorch.org/docs/stable/elastic/run.html (torchrun)
- https://docs.pytorch.org/docs/stable/distributed.html
- https://docs.pytorch.org/docs/stable/notes/ddp.html

Typical usage in run.py:
    from src.utils.device import get_device
    from src.utils.distributed import init_if_enabled, is_main_process, cleanup

    device = get_device(cfg["device"])
    ddp_on = init_if_enabled(cfg["ddp"], device)
    # ... build model, move to device
    if ddp_on:
        model = DDP(model, device_ids=[device.index] if device.type=="cuda" else None)
    # ... training/eval loops
    cleanup()
"""

import os
from datetime import timedelta

import torch
import torch.distributed as dist


def init_if_enabled(
    ddp_cfg: dict,
    device: torch.device,
) -> bool:
    """
    Initialize torch.distributed process group if enabled.

    Args:
        ddp_cfg: dict with keys:
            - "enabled": bool
            - "backend": "nccl" | "gloo" | "mpi" | None (auto-selected)
            - "timeout_sec": int (default 1800)
        device: torch.device (used to auto-select backend if not specified)

    Returns:
        True if DDP was initialized in this process, False otherwise.

    Environment variables (set by torchrun):
        - RANK: global rank
        - LOCAL_RANK: local rank (GPU index on this node)
        - WORLD_SIZE: total number of processes

    Note:
        If device is CUDA, torch.cuda.set_device(LOCAL_RANK) should already
        have been called by get_device() for proper DDP operation.
    """
    if not ddp_cfg.get("enabled", False):
        return False

    # Verify torchrun envs exist
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "DDP enabled but RANK/WORLD_SIZE not set. "
            "Did you launch with torchrun? "
            "Example: torchrun --nproc_per_node=2 src/run.py --config configs/v3.yaml"
        )

    # Select backend
    backend = ddp_cfg.get("backend")
    if backend is None:
        # Auto-select: nccl for CUDA, gloo for CPU/MPS
        backend = "nccl" if device.type == "cuda" else "gloo"

    # Timeout
    timeout_sec = ddp_cfg.get("timeout_sec", 1800)
    timeout = timedelta(seconds=timeout_sec)

    # Initialize process group
    # init_method="env://" reads RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT from env
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=timeout,
    )

    return True


def is_main_process() -> bool:
    """
    Check if current process is main (rank 0).

    Returns True when distributed is not initialized (treat single-process as main).
    """
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """
    Get global rank of current process.

    Returns 0 when distributed is not initialized (single-process mode).
    """
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """
    Get total number of processes.

    Returns 1 when distributed is not initialized (single-process mode).
    """
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_local_rank() -> int:
    """
    Get local rank (GPU index on this node) from environment.

    Returns 0 when LOCAL_RANK not set (single-process mode).
    """
    return int(os.environ.get("LOCAL_RANK", 0))


def barrier() -> None:
    """
    Synchronize all processes.

    No-op when distributed is not initialized.
    """
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup() -> None:
    """
    Destroy process group.

    Call once at end of main() to properly tear down distributed state.
    No-op when distributed is not initialized.
    """
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def ddp_reduce_mean(
    tensor: torch.Tensor,
    group: torch.distributed.ProcessGroup | None = None,
) -> torch.Tensor:
    """
    Average a tensor across all ranks.

    No-op when distributed is not initialized. Returns the input tensor
    instance after the reduction so it can be chained in-place.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("ddp_reduce_mean expects a torch.Tensor input")

    if not dist.is_available() or not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size(group=group)
    if world_size < 2:
        return tensor

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    tensor.div_(world_size)
    return tensor
