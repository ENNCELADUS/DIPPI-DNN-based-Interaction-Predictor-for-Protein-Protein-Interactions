"""
Device selection and management utilities.

Provides deterministic device picking plus a lightweight DeviceManager that
handles tensor/device transfers and optional DDP wrapping for trainers.
"""

from __future__ import annotations

import os
import random
from typing import Any, Mapping, Union, Optional

import numpy as np
import torch


def get_device(
    device_cfg: Union[str, dict],
    prefer_local_rank: bool = True,
) -> torch.device:
    """
    Deterministically select a torch.device from config.

    Args:
        device_cfg: Either a string ("cuda", "mps", "cpu", "auto") or dict:
            {
                "strategy": "auto" | "cuda" | "mps" | "cpu",
                "gpu_id": int or None  # optional explicit GPU index
            }
        prefer_local_rank: If True and CUDA selected, prefer LOCAL_RANK env
            over gpu_id for torchrun compatibility.

    Returns:
        torch.device instance.
    """
    # Normalize input to dict
    if isinstance(device_cfg, str):
        cfg = {"strategy": device_cfg, "gpu_id": None}
    else:
        cfg = device_cfg.copy()

    strategy = cfg.get("strategy", "auto")
    gpu_id = cfg.get("gpu_id", None)

    # Resolve strategy to concrete device type
    if strategy == "auto":
        if torch.cuda.is_available():
            device_type = "cuda"
        elif torch.backends.mps.is_available():
            device_type = "mps"
        else:
            device_type = "cpu"
    else:
        device_type = strategy

    # Build device
    if device_type == "cuda":
        # Prefer LOCAL_RANK for torchrun multi-GPU
        local_rank_env = os.environ.get("LOCAL_RANK")
        if prefer_local_rank and local_rank_env is not None:
            idx = int(local_rank_env)
        elif gpu_id is not None:
            idx = gpu_id
        else:
            idx = 0  # default to cuda:0

        device = torch.device(f"cuda:{idx}")
        torch.cuda.set_device(idx)

    elif device_type == "mps":
        device = torch.device("mps")

    elif device_type == "cpu":
        device = torch.device("cpu")

    else:
        raise ValueError(
            f"Unknown device strategy: {device_type}. "
            "Must be one of: auto, cuda, mps, cpu."
        )

    return device


class DeviceManager:
    """
    Lightweight helper for device selection, seeding, and batch transfers.

    Trainers interact with DeviceManager instead of importing torch.cuda
    utilities directly. The manager can optionally wrap models with DDP when
    `use_ddp=True` and a process group is initialized.
    """

    def __init__(
        self,
        *,
        device_cfg: Union[str, dict, torch.device, None] = None,
        prefer_gpu: bool = True,
        use_ddp: bool = False,
        prefer_local_rank: bool = True,
    ) -> None:
        self.prefer_gpu = prefer_gpu
        self.use_ddp = use_ddp
        self.prefer_local_rank = prefer_local_rank

        if isinstance(device_cfg, torch.device):
            self._device: Optional[torch.device] = device_cfg
            self._device_cfg: Union[str, dict, None] = None
        else:
            self._device = None
            if device_cfg is not None:
                self._device_cfg = device_cfg
            else:
                self._device_cfg = (
                    {"strategy": "cuda"} if prefer_gpu else {"strategy": "cpu"}
                )

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def select_device(self) -> torch.device:
        """Return cached torch.device, selecting it on first call."""
        if self._device is None:
            cfg = self._device_cfg or (
                {"strategy": "cuda"} if self.prefer_gpu else {"strategy": "cpu"}
            )
            self._device = get_device(cfg, prefer_local_rank=self.prefer_local_rank)
        return self._device

    def set_seed(
        self,
        seed: int,
        *,
        deterministic: bool = False,
        logger: Optional[Any] = None,
    ) -> None:
        """Seed RNGs for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if logger is not None:
            logger.info(f"Random seed set to {seed} (deterministic={deterministic})")

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Move model to managed device and optionally wrap with DDP.
        """
        device = self.select_device()
        model = model.to(device)

        if (
            self.use_ddp
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            from torch.nn.parallel import DistributedDataParallel as DDP

            if not isinstance(model, DDP):
                ddp_kwargs = {}
                if device.type == "cuda":
                    device_ids = [device.index] if device.index is not None else None
                    if device_ids is not None:
                        ddp_kwargs["device_ids"] = device_ids
                        ddp_kwargs["output_device"] = device.index
                model = DDP(model, **ddp_kwargs)

        return model

    def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Return underlying module if wrapped with DDP."""
        return model.module if hasattr(model, "module") else model

    def to_device(
        self,
        batch: Any,
        device: Optional[torch.device] = None,
    ) -> Any:
        """Recursively move tensors in `batch` to target device."""
        target = device or self.select_device()

        if isinstance(batch, torch.Tensor):
            return batch.to(target, non_blocking=True)

        if isinstance(batch, Mapping):
            return {k: self.to_device(v, target) for k, v in batch.items()}

        if isinstance(batch, tuple):
            return tuple(self.to_device(v, target) for v in batch)

        if isinstance(batch, list):
            return [self.to_device(v, target) for v in batch]

        if hasattr(batch, "to") and callable(getattr(batch, "to")):
            try:
                return batch.to(target)
            except Exception:
                return batch

        return batch


def is_cuda(device: torch.device) -> bool:
    """Check if device is CUDA."""
    return device.type == "cuda"


def is_mps(device: torch.device) -> bool:
    """Check if device is MPS (Apple Metal)."""
    return device.type == "mps"


def is_cpu(device: torch.device) -> bool:
    """Check if device is CPU."""
    return device.type == "cpu"
