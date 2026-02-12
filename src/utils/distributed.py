"""Distributed helpers for DDP-capable orchestration."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DistributedContext:
    """Process metadata for distributed execution.

    Attributes:
        ddp_enabled: Whether DDP was requested in config.
        is_distributed: Whether process-group is initialized.
        rank: Global process rank.
        local_rank: Local process rank on node.
        world_size: Number of processes in the job.
    """

    ddp_enabled: bool
    is_distributed: bool
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_main_process(self) -> bool:
        """Return whether this process is the main rank."""
        return self.rank == 0


def initialize_distributed(ddp_enabled: bool) -> DistributedContext:
    """Initialize distributed context and return process metadata.

    Args:
        ddp_enabled: Whether distributed mode is enabled in config.
    """
    if not ddp_enabled:
        return DistributedContext(ddp_enabled=False, is_distributed=False)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        LOGGER.warning("DDP enabled but WORLD_SIZE<=1; running single-process mode.")
        return DistributedContext(ddp_enabled=True, is_distributed=False)

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    LOGGER.info(
        "Initialized distributed process group (backend=%s rank=%d local_rank=%d world_size=%d).",
        backend,
        rank,
        local_rank,
        world_size,
    )
    return DistributedContext(
        ddp_enabled=True,
        is_distributed=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )


def distributed_barrier(context: DistributedContext) -> None:
    """Synchronize processes when distributed mode is active.

    Args:
        context: Distributed process metadata.
    """
    if context.is_distributed and dist.is_initialized():
        backend = dist.get_backend()
        if backend == "nccl" and torch.cuda.is_available():
            dist.barrier(device_ids=[context.local_rank])
            return
        dist.barrier()


def cleanup_distributed(context: DistributedContext) -> None:
    """Tear down distributed process group if initialized.

    Args:
        context: Distributed process metadata.
    """
    if context.is_distributed and dist.is_initialized():
        dist.destroy_process_group()
